import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
from functools import partial
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling

class MyCLIPVisionModel(CLIPVisionModel):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        out = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return {'last_feat': out.last_hidden_state.permute(0, 2, 1)}


class MyCLIPVisionModelWithProjection(CLIPVisionModelWithProjection):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)
        # last_hidden_state = vision_outputs.last_hidden_state
        # hidden_states = vision_outputs.hidden_states
        # attentions = vision_outputs.attentions

        return {'last_feat': image_embeds.unsqueeze(-1)}

class MyCLIPTextModel(nn.Module):
    
    def __init__(self, pretrained_model_path, learnable_token_len=5):
        super().__init__()
        self.clip = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.learnable_token_len = learnable_token_len
        self.learnable_embedding = nn.Parameter(torch.zeros(self.learnable_token_len, 768), requires_grad=True)
        self.register_parameter("learnable_embedding", self.learnable_embedding)
        self.EOSIDX = 49407
    
    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None) -> torch.Tuple | BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.clip.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.clip.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.clip.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # index every EOS
        eos_indexes = torch.argmax((input_ids == self.EOSIDX).float(), dim=1)
        batch_indexes = torch.arange(input_shape[0])
        assert len(eos_indexes) == input_shape[0]
    
        hidden_states = self.clip.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)
        # save EOS embedding
        eos_embedding = hidden_states[batch_indexes, eos_indexes, :].clone()
        # relace EOS and padding embedding with soft prompting 
        for idx, eos_idx in enumerate(eos_indexes):
            if eos_idx+self.learnable_token_len < 77:
                hidden_states[idx, eos_idx:eos_idx+self.learnable_token_len, :] = self.learnable_embedding
            else:
                hidden_states[idx, eos_idx:, :] = self.learnable_embedding
                hidden_states[idx, -1, :] = eos_embedding
                print(f"prompt too long, trunct {eos_idx+self.learnable_token_len-77} tokens")
        # # concat hidden states with learnable prompts
        # # hidden_states: (b 77 768) -> (b c+77 768)
        # hidden_states = torch.cat([self.learnable_embedding.unsqueeze(0).repeat(input_shape[0], 1, 1), hidden_states], dim=1)
        # update input shape
        input_shape = hidden_states.size()[:2]
        
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        
        encoder_outputs = self.clip.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip.text_model.final_layer_norm(last_hidden_state)

        if self.clip.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.clip.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
def get_dino_image_descriptor_model():
    raise NotImplementedError('Not yet supported')


def get_mae_image_descriptor_model():
    raise NotImplementedError('Not yet supported')


def get_image_descriptor_model(descriptor_name, pretrained_model_path, unet):
    text_encoder = tokenizer = image_descriptor_model = None
    if descriptor_name == 'clip_image':
        # image_descriptor_model = MyCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        image_descriptor_model = MyCLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        unet.modify_encoder_hidden_state_proj(1024, 768)

    elif descriptor_name == 'clip_image_proj':
        # image_descriptor_model = MyCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        image_descriptor_model = MyCLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

    elif descriptor_name == 'dino_image':
        raise NotImplementedError('DINO is not yet supported')
        get_dino_image_descriptor_model()
        unet.modify_encoder_hidden_state_proj(768, 768)
        print('adding linear projection to unet for image descriptors')

    elif descriptor_name == 'mae':
        raise NotImplementedError('MAE is not yet supported')
        get_mae_image_descriptor_model()
        unet.modify_encoder_hidden_state_proj(768, 768)
        print('adding linear projection to unet for image descriptors')

    elif descriptor_name == 'learnable':
        unet.define_learnable_embeddings(128, 768)
        print(f'Successfully added learnable object queries to unet as {unet.object_queries}')

    elif descriptor_name == 'remove':
        unet.remove_cross_attention()
        print('Successfully removed cross attention layers from unet')
    elif descriptor_name == 'sp_clip':
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = MyCLIPTextModel(pretrained_model_path=pretrained_model_path, learnable_token_len=20)
        for n, p in text_encoder.named_parameters():
            p.requires_grad_(True)
        print('Succesfully loaded soft prompting CLIP text encoder')
    else:
        assert descriptor_name == 'none'
        # load the pretrained CLIP model
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        print('Succesfully loaded pretrained CLIP text encoder')

    return image_descriptor_model, text_encoder, tokenizer
