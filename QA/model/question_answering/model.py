import os
import sys
import argparse
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel,AutoModelForCausalLM
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name, llm_layer

class QAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(QAModel, self).__init__()
        self.args = args

        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(huggingface_model_name)
        if args.model_ispretrained:
            self.model = AutoModel.from_pretrained(huggingface_model_name, cache_dir=self.args.cache_path)
        else:
            self.model = AutoModel.from_config(self.config)
        self.hidden_size = self.model.config.hidden_size

        if self.args.method == 'base_llm':
            llm_model_name = get_huggingface_model_name(self.args.llm_model)
            self.llm_layer, self.llm_embed_size, self.llm_hidden_size = llm_layer(llm_model_name, args)        
            self.llama_dim_mapper1 = nn.Linear(self.hidden_size, self.llm_embed_size, bias=False)
            self.llama_dim_mapper2 = nn.Linear(self.llm_embed_size, self.hidden_size, bias=False)

        self.start_classifier = nn.Linear(self.hidden_size, 1)
        self.end_classifier = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> tuple:
        device = input_ids.device
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        sequence_output = model_output.last_hidden_state

        if self.args.method == 'base_llm':
            batch_size, seq_length, _ = sequence_output.size()
            
            attention_mask = torch.ones((batch_size, seq_length), dtype=sequence_output.dtype).to(device)  

            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

            sequence_output = self.llama_dim_mapper1(sequence_output)
            if self.args.llm_model == 'falcon':
                llm_outputs = self.llm_layer(
                                    hidden_states=sequence_output,
                                    attention_mask=attention_mask[:, None, None, :],  
                                    position_ids=position_ids,
                                    past_key_value=None,
                                    output_attentions=None,
                                    use_cache=None,
                                    alibi=False
                                    )
            else:
                llm_outputs = self.llm_layer(
                                    hidden_states=sequence_output,
                                    attention_mask=attention_mask[:, None, None, :],
                                    position_ids=position_ids,
                                    past_key_value=None,
                                    output_attentions=None,
                                    use_cache=None,
                                    )
            sequence_output = llm_outputs[0]
            sequence_output = self.llama_dim_mapper2(sequence_output)

        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)

        start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
        end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
        
        return start_logits, end_logits
