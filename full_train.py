from bert_pretrain import *
from rl_train import *
from reward_estimator import *
from data_preprocess import *
import argparse
from trl.gpt2 import GPT2HeadWithValueModel
from transformers import GPT2Tokenizer, BertTokenizer
%matplotlib inline

word_embed_path = "word_embed.kv"

models_output_dir = 'models/'

gpt2_model_dir = "gpt2_model/"
bert_model_dir = "bert_model/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        default="/datasets/gyafc.csv",
                        type=str,
                        help="The path to dataset for training.")
    parser.add_argument("--formal_to_informal_path",
                        default=None,
                        type=str,
                        help="The path to pretrained Formal-To-Informal models.")
    parser.add_argument("--informal_to_formal_path",
                        default=None,
                        type=str,
                        help="The path to pretrained Informal-To-Formal models.")
    parser.add_argument("--output_dir",
                        default='/models/',
                        type=str,
                        help="The output directory where the models will be saved after training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()



    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab = list(tokenizer.encoder.keys())
    assert(len(vocab) == tokenizer.vocab_size)
    # save word vectors for semantic critic
    save_word_embeddings(vocab, 0, word_embed_path)

    # CREATE FORMAL-TO-INFORMAL MODELS

    gpt2_fi_model_path = os.path.join(args.formal_to_informal_path, gpt2_model_dir)
    bert_fi_model_path = os.path.join(args.formal_to_informal_path, bert_model_dir)
    gpt2_fi_out_model_path = os.path.join(args.output_dir, gpt2_model_dir)
    bert_fi_out_model_path = os.path.join(args.output_dir, bert_model_dir)

    gpt2_fi_model_name = 'gpt2' if not os.path.exists(gpt2_model_path) else gpt2_model_path
    gpt2_fi_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_fi_model_name)
    gpt2_fi_model = GPT2HeadWithValueModel.from_pretrained(gpt2_fi_model_name)
    gpt2_fi_model.cuda()

    bert_fi_model_name = 'bert-base-cased' if not os.path.exists(bert_model_path) else bert_model_path
    bert_fi_tokenizer = BertTokenizer.from_pretrained(bert_fi_model_name, do_lower_case=False)
    bert_fi_model = CreateBertModel(bert_fi_model_name)
    bert_fi_model.cuda()

    # DATA PREPROCESSING

    train_fi_dataloader, validaton_fi_dataloader, prediction_fi_dataloader = data_preprocess(dataset_path, bert_fi_tokenizer)

    # BERT PRETRAINING


    bert_pretrain(bert_fi_model, bert_fi_out_model_path, train_dataloader, validation_dataloader)

    # ADVERSARIAL TRAINING

    reward_estimator = RewardEstimator(FIStyleCritic(bert_fi_model), SemanticCritic(word_embed_path))
    rl_train(gpt2_fi_model, bert_fi_model, gpt2_fi_tokenizer, bert_fi_tokenizer, gpt2_fi_out_model_path, bert_fi_out_model_path, train_fi_dataloader, reward_estimator)
    rl_eval(gpt2_fi_model, gpt2_fi_tokenizer, bert_fi_tokenizer, validation_fi_dataloader, reward_estimator)

    # CREATE INFORMAL-TO-FORMAL MODELS

    gpt2_if_model_path = os.path.join(args.informal_to_formal_path, gpt2_model_dir)
    bert_if_model_path = os.path.join(args.informal_to_formal_path, bert_model_dir)
    gpt2_if_out_model_path = os.path.join(args.output_dir, gpt2_model_dir)
    bert_if_out_model_path = os.path.join(args.output_dir, bert_model_dir)

    gpt2_if_model_name = 'gpt2' if not os.path.exists(gpt2_if_model_path) else gpt2_if_model_path
    gpt2_if_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_if_model_name)
    gpt2_if_model = GPT2HeadWithValueModel.from_pretrained(gpt2_if_model_name)
    gpt2_if_model.cuda()

    bert_if_model_name = 'bert-base-cased' if not os.path.exists(bert_if_model_path) else bert_if_model_path
    bert_if_tokenizer = BertTokenizer.from_pretrained(bert_if_model_name, do_lower_case=False)
    bert_if_model = CreateBertModel(bert_if_model_name)
    bert_if_model.cuda()

    # DATA PREPROCESSING

    train_if_dataloader, validaton_if_dataloader, prediction_if_dataloader = data_preprocess(dataset_path, bert_if_tokenizer)

    # BERT PRETRAINING

    bert_pretrain(bert_if_model, bert_if_out_model_path, train_dataloader, validation_dataloader)

    # ADVERSARIAL TRAINING

    reward_estimator = RewardEstimator(IFStyleCritic(bert_if_model), SemanticCritic(word_embed_path))
    rl_train(gpt2_if_model, bert_if_model, gpt2_if_tokenizer, bert_if_tokenizer, gpt2_if_out_model_path, bert_if_out_model_path, train_if_dataloader, reward_estimator)
    rl_eval(gpt2_if_model, gpt2_if_tokenizer, bert_if_tokenizer, validation_if_dataloader, reward_estimator)
