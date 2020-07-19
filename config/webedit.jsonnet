local embedding_dim = 300;
local encoder_size = 300;
local decoder_size = 600;
local num_layers = 2;
local batch_size = 512;
local num_epochs = 50;
local patience = 50;
local max_decoding_steps = 80;
local beam_size = 4;
local learning_rate = 1e-3;
local grad_clipping = 5.0;
local validation_metric = "+BLEU";
local lazy = true;
local cuda_device = 0;

{
    "train_data_path": "./data/webedit/train.jsonl",
    "validation_data_path": "./data/webedit/dev.jsonl",
    "dataset_reader": {
        "type": "edit-reader",
        "lazy": lazy
    },
    "model": {
        "type": "editor",
        "embed": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder_size": encoder_size,
        "decoder_size": decoder_size,
        "num_layers": num_layers,
        "beam_size": beam_size,
        "max_decoding_steps": max_decoding_steps,
    },
    "iterator": {
        "type": "bucket",

        "sorting_keys": [["action_tokens", "num_tokens"]],
        "batch_size": batch_size,
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "adam",
            "lr": learning_rate,
            "amsgrad": true
        },
        "patience": patience,
        "validation_metric": validation_metric,
        "grad_clipping": grad_clipping,
        "cuda_device": cuda_device
    }
}
