## acknowledgement
We use the paper《Leveraging passage retrieval with generative models for open domain question answering》by G Izacard, E Grave, etc. as our code base.

Our solution is a three-stage method. (1) Retriever (2) Reader (3) Generator. More details can be seen in our workshop paper 《G4: Grounding-guided Goal-oriented Dialogues Generation with Multiple Documents》

## Dependencies
- Python 3
- [PyTorch](http://pytorch.org/) (1.6 or 1.9, i have try 1.6 in V100 and 1.9 in A100)
- [Transformers](http://huggingface.co/transformers/) (3.0.2, unlikely to work with a different version)
- datasets(1.16.1 for evaluation em, f1, bleu, meteor, rouge)
- nltk(for evaluation em, f1, bleu, meteor, rouge)
- sacrebleu(for bleu)
`pip install datasets==1.16.1 transformers==3.0.2 nltk rouge-score  sacrebleu`


# Data
### Preprocess data

Retrieval id is needed from DPR before Preprocess(We upload the retrieval result in temp. We will upload the code of retriever in future!)
```shell
python data_preprocess_fid.py
```

### Data format

The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `id`: example id, optional
- `question`: question text
- `target`: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
- `answers`: list of answer text for evaluation, also used for training if target is not given
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
        - `text`: passage text

Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```

# Fusion-in-Decoder

Fusion-in-Decoder models can be trained using [`train_reader.py`](train_reader.py) and evaluated with [`test_reader.py`](test_reader.py).

### Train

[`train_reader.py`](train_reader.py) provides the code to train a model. An example usage of the script is given below:

train
```shell
python -m torch.distributed.launch --nproc_per_node=4 --master_port 5678 train_reader.py \
        --train_data ./data/train_50.json \
        --eval_data ./data/val_50.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --accumulation_steps 2 \
        --n_context 50 \
        --name my_experiment \
        --text_maxlength 512 \
        --answer_maxlength 50 \
        --total_steps 50000 \
        --use_checkpoint \
```

### Test

You can evaluate your model or a pretrained model with [`test_reader.py`](test_reader.py). An example usage of the script is provided below.

```shell
python test_reader.py \
        --model_path ./checkpoint/my_experiment/checkpoint/best_dev \
        --eval_data .data/val_50.json \
        --per_gpu_batch_size 16 \
        --n_context 50 \
        --text_maxlength 512 \
        --answer_maxlength 50 \
        --name val_n_50 \
        --write_results
```

### Add grounding

adding grounding information in passages will enhance the representation of passages.
```shell
# need train_grounding.json, val_grounding.json. We get them by a simple span-extraction MRC model. Train model with train data, and infer grounding in train and val data.(We upload the grounding result in temp. We will upload the code of reader in future!)
python add_grounding_sp_window.py
# generate the train and val data with enhanced passages
# train_50_w_grounding_sp_window.json, val_50_w_grounding_sp_window.json
python -m torch.distributed.launch --nproc_per_node=4 --master_port 5678 train_reader.py \
        --train_data ./data/train_50_w_grounding_sp_window.json \
        --eval_data ./data/val_50_w_grounding_sp_window.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --accumulation_steps 2 \
        --n_context 50 \
        --name my_experiment \
        --text_maxlength 512 \
        --answer_maxlength 50 \
        --total_steps 50000 \
        --use_checkpoint \
```
