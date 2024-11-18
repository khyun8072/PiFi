import numpy as np
import os
import math
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name, llm_layer

def shift_right(input_ids, pad_token_id):
    """Shift the input_ids to the right by inserting a bos token at the beginning."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id  
    return shifted_input_ids

class TranslationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(TranslationModel, self).__init__()
        self.args = args

        model_name = get_huggingface_model_name(args.model_type)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.embed_size = self.model.config.d_model
        self.hidden_size = self.model.config.d_model

        if self.args.method == 'base_llm':
            llm_model_name = get_huggingface_model_name(self.args.llm_model)
            self.llm_layer, self.llm_embed_size, self.llm_hidden_size = llm_layer(llm_model_name, args)
            self.llama_dim_mapper1 = nn.Linear(self.embed_size, self.llm_embed_size, bias=False)
            self.llama_dim_mapper2 = nn.Linear(self.llm_embed_size, self.embed_size, bias=False)

    def forward(self, src, src_attention_mask, tgt, tgt_attention_mask):
        if self.args.method == 'base_llm':
            if self.args.model_type == 't5':
                encoder_outputs = self.model.encoder(input_ids=src, attention_mask=src_attention_mask, return_dict=True)
            elif self.args.model_type == 'bart':
                encoder_outputs = self.model.model.encoder(input_ids=src, attention_mask=src_attention_mask, return_dict=True)
            hidden_states = encoder_outputs.last_hidden_state

            hidden_states = hidden_states.to(src.device)
            batch_size = hidden_states.size(0)
            seq_length = hidden_states.size(1)

            attention_mask = torch.ones((batch_size, seq_length), dtype=hidden_states.dtype).to(src.device)
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(src.device)

            hidden_states = self.llama_dim_mapper1(hidden_states)
            if self.args.llm_model == 'falcon':
                llm_outputs = self.llm_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask[:, None, None, :],  
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                    alibi=False
                )
            else:
                llm_outputs = self.llm_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask[:, None, None, :],  
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                )
            
            llm_outputs = llm_outputs[0].squeeze(1)
            hidden_states = self.llama_dim_mapper2(llm_outputs)
            if self.args.model_type == 't5':
                tgt = self.model._shift_right(tgt)
            elif self.args.model_type == 'bart':
                tgt = shift_right(tgt, self.model.config.decoder_start_token_id)
            if self.args.model_type == 't5':
                decoder_outputs = self.model.decoder(input_ids=tgt, attention_mask=tgt_attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=src_attention_mask)
            elif self.args.model_type == 'bart':
                decoder_outputs = self.model.model.decoder(input_ids=tgt, attention_mask=tgt_attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=src_attention_mask)
            
            logits = self.model.lm_head(decoder_outputs.last_hidden_state)

            return logits
        
        else:
            if self.args.model_type == 't5':
                tgt = self.model._shift_right(tgt)
            elif self.args.model_type == 'bart':
                tgt = shift_right(tgt, self.model.config.decoder_start_token_id)
            output = self.model(input_ids=src, attention_mask=src_attention_mask, decoder_input_ids=tgt, decoder_attention_mask=tgt_attention_mask)
            logits = output.logits
            return logits

    def generate(self, src, src_attention_mask, max_length):
        if self.args.method == 'base_llm':
            if self.args.model_type == 't5':
                encoder_outputs = self.model.encoder(input_ids=src, attention_mask=src_attention_mask, return_dict=True)
            elif self.args.model_type == 'bart':
                encoder_outputs = self.model.model.encoder(input_ids=src, attention_mask=src_attention_mask, return_dict=True)
            
            hidden_states = encoder_outputs.last_hidden_state

            hidden_states = hidden_states.to(src.device)
            batch_size = hidden_states.size(0)
            seq_length = hidden_states.size(1)


            attention_mask = torch.ones((batch_size, seq_length), dtype=hidden_states.dtype).to(src.device)
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(src.device)


            hidden_states = self.llama_dim_mapper1(hidden_states)

            if self.args.llm_model == 'falcon':
                llm_outputs = self.llm_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask[:, None, None, :],  
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                    alibi=False
                )
            else:
                llm_outputs = self.llm_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask[:, None, None, :],  
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                )
            
            
            llm_outputs = llm_outputs[0]
            hidden_states = self.llama_dim_mapper2(llm_outputs)
            num_beams=5

            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=src.device)
            beam_scores[:, 1:] = -1e9  
            beam_scores = beam_scores.view(-1)  
            generated_ids = torch.full((batch_size * num_beams, 1), self.model.config.decoder_start_token_id, dtype=torch.long).to(src.device)
            eos_flags = torch.zeros(batch_size * num_beams, dtype=torch.bool).to(src.device)  
            for cur_len in range(max_length):
                if eos_flags.all():  
                    break
                
                if self.args.model_type == 't5':
                    decoder_outputs = self.model.decoder(
                        input_ids=generated_ids, 
                        encoder_hidden_states=hidden_states.repeat_interleave(num_beams, dim=0),
                        encoder_attention_mask=src_attention_mask.repeat_interleave(num_beams, dim=0)
                    )
                elif self.args.model_type == 'bart':
                    decoder_outputs = self.model.model.decoder(
                        input_ids=generated_ids, 
                        encoder_hidden_states=hidden_states.repeat_interleave(num_beams, dim=0),
                        encoder_attention_mask=src_attention_mask.repeat_interleave(num_beams, dim=0)
                    )
                logits = self.model.lm_head(decoder_outputs.last_hidden_state)
                
                next_token_logits = logits[:, -1, :]  
                scores = torch.log_softmax(next_token_logits, dim=-1) 
                next_scores = scores + beam_scores[:, None].expand_as(scores)  

                next_scores = next_scores.view(batch_size, num_beams * scores.size(-1))
                next_token_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

                next_batch_beam = []

                for batch_idx in range(batch_size):
                    next_sent_beam = []

                    for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_token_scores[batch_idx])
                    ):
                        beam_id = beam_token_id // scores.size(-1)
                        token_id = beam_token_id % scores.size(-1)

                        effective_beam_id = batch_idx * num_beams + beam_id

                        if token_id == self.model.config.eos_token_id:
                            eos_flags[effective_beam_id] = True 
                            if len(next_sent_beam) < num_beams:
                                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                            continue  

                        if len(next_sent_beam) == num_beams:
                            break

                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    next_batch_beam.extend(next_sent_beam)

                beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
                beam_tokens = generated_ids.new([x[1] for x in next_batch_beam])
                beam_idx = generated_ids.new([x[2] for x in next_batch_beam])

                generated_ids = generated_ids[beam_idx, :]
                generated_ids = torch.cat([generated_ids, beam_tokens.unsqueeze(1)], dim=-1)
            final_sequences = []
            for batch_idx in range(batch_size):
                best_seq = generated_ids[batch_idx * num_beams]
                final_sequences.append(best_seq)
            return torch.stack(final_sequences)
        else:
            gen = self.model.generate(input_ids=src, attention_mask=src_attention_mask, max_length=max_length, num_beams=5)
            return gen



    
