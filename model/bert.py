from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

Class Bert(torch.nn.Module):
    def __init__(self, args):
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            args.model,
            num_labels = args.num_labels,
            finetuning_task = args.task_name,
            cache_dir = './'+args.model+'/pretrained_model',
            revision = 'main',
            use_auth_token = None,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            from_tf = bool(".ckpt" in args.model),
            config = config,
            cache_dir = './'+args.model+'/pretrained_model',
            revision = 'main',
            use_auth_token = None,
            ignore_mismatched_sizes = False,
        )

        self.model.config.label2id = {l: i for i, l in enumerate(args.label_list)}
        self.model.config.id2label = {id: label for label, id in config.label2id.items()}

    def forward(self, **kwargs):
        input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict = kwargs
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
