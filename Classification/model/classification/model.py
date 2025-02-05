import os
import sys
import argparse
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name, llm_layer

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(huggingface_model_name, cache_dir=self.args.cache_path)
        if args.model_ispretrained:
            self.model = AutoModel.from_pretrained(huggingface_model_name, cache_dir=self.args.cache_path)
        else:
            self.model = AutoModel.from_config(self.config)
        self.embed_size = self.model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = self.args.num_classes

        if self.args.method == 'base_llm':
            llm_model_name = get_huggingface_model_name(self.args.llm_model)
            self.llm_layer, self.llm_embed_size, self.llm_hidden_size = llm_layer(llm_model_name, args)        
            self.llama_dim_mapper1 = nn.Linear(self.embed_size, self.llm_embed_size, bias=False)
            self.llama_dim_mapper2 = nn.Linear(self.llm_embed_size, self.embed_size, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.Dropout(self.args.dropout_rate),
            nn.GELU(),
            nn.Linear(self.embed_size, self.num_classes),
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        if self.args.model_type == 'modern_bert' or self.args.model_type == 'smollm':
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            
        else:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
            
        if self.args.padding == 'cls':
            cls_output = model_output.last_hidden_state[:, 0, :]

        elif self.args.padding == 'average_pooling_with_padding':
            cls_output = torch.mean(model_output.last_hidden_state, dim=1)  

        elif self.args.padding == 'average_pooling_without_padding':
            sum_output = torch.sum(model_output.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
            count_non_padding = torch.sum(attention_mask, dim=1, keepdim=True)
            cls_output = sum_output / count_non_padding


        if self.args.method == 'base':
            classification_logits = self.classifier(cls_output)
        elif self.args.method == 'base_llm':
            cls_output = cls_output.unsqueeze(1).to(device)
            batch_size = cls_output.size(0)
            seq_length = cls_output.size(1)
            attention_mask = torch.ones((batch_size, seq_length), dtype=cls_output.dtype).to(device)  
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(device)
            # position_embeddings 생성
            cls_output = self.llama_dim_mapper1(cls_output)
            if self.args.llm_model == 'falcon':
                llm_outputs = self.llm_layer(
                                    hidden_states=cls_output,
                                    attention_mask=attention_mask[:, None, None, :], 
                                    position_ids=position_ids,
                                    past_key_value=None,
                                    output_attentions=None,
                                    use_cache=None,
                                    alibi=False
                                    )
            else:
                llm_outputs = self.llm_layer(
                                    hidden_states=cls_output,
                                    attention_mask=attention_mask[:, None, None, :],
                                    position_ids=position_ids,
                                    past_key_value=None,
                                    output_attentions=None,
                                    use_cache=None,
                                    )
            llm_outputs = llm_outputs[0].squeeze(1)
            llm_outputs = self.llama_dim_mapper2(llm_outputs)
            classification_logits = self.classifier(llm_outputs) 
            
        
        return classification_logits


    