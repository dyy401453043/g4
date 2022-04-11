import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

train_data = []
with open('./data/train_50.json','r') as f:
	train_data = json.load(f)

a, b = 0, 0
cq0, cq1 = 0, 0
count0,count1 = 0, 0
with open('./data/grounding/train_grounding_100.json','r') as f:
	grounding_dict = json.load(f)
	for data in tqdm(train_data):
		id = data['id']
		t = tokenizer.tokenize(data['question'].replace('[SEP]', '<extra_id_99>'))
		if len(t) > 200:
			cq0 += 1
			data['question'] = tokenizer.convert_tokens_to_string(t[:200])
		else:
			cq1 += 1
			data['question'] = tokenizer.convert_tokens_to_string(t)
		for j, ctx in enumerate(data['ctxs'][:50]):
			grounding = grounding_dict[id+f'_{j}']
			grounding = grounding.strip().replace('\n', '').replace('  ',' ')
			passage = ctx['text'].replace('  ',' ').replace('  ',' ')
			if (grounding in passage) and grounding:
				index = passage.find(grounding)
				p1, p2, p3 = passage[:index], passage[index:index+len(grounding)], passage[index+len(grounding):]
				t1, t2, t3 = tokenizer.tokenize(p1), tokenizer.tokenize(p2), tokenizer.tokenize(p3)
				if len(t1) + len(t2) + len(t3) < 300:
					passage = tokenizer.convert_tokens_to_string(t1) + ' <extra_id_0> ' + tokenizer.convert_tokens_to_string(t2) + ' <extra_id_1> ' + tokenizer.convert_tokens_to_string(t3)
					count0+=1
				else:
					temp = int((300 - len(t2)) / 2)
					passage = tokenizer.convert_tokens_to_string(t1[len(t1)-temp:]) + ' <extra_id_0> ' + tokenizer.convert_tokens_to_string(t2) + ' <extra_id_1> ' + tokenizer.convert_tokens_to_string(t3[:temp])
					count1+=1
				b += 1
			else:
				a += 1
			ctx['text'] = passage
print(a)
print(b)
print(f"cq0:{cq0}, cq1:{cq1}")
print(f"count0:{count0}, count1:{count1}")

with open('./data/grounding/train_50_sp_window.json', 'w') as f:
        json.dump(train_data, f, indent=4)


val_data = []
with open('./data/val_50.json','r') as f:
	val_data = json.load(f)

c, d = 0, 0
with open('./data/grounding/val_grounding_100.json','r') as f:
	grounding_dict = json.load(f)
	for data in val_data:
		id = data['id']
		t = tokenizer.tokenize(data['question'].replace('[SEP]', '<extra_id_99>'))
		if len(t) > 200:
			data['question'] = tokenizer.convert_tokens_to_string(t[:200])
		else:
			data['question'] = tokenizer.convert_tokens_to_string(t)
		for j, ctx in enumerate(data['ctxs'][:50]):
			grounding = grounding_dict[id+f'_{j}']
			grounding = grounding.strip().replace('\n', '').replace('  ',' ')
			passage = ctx['text'].replace('  ',' ').replace('  ',' ')
			if (grounding in passage) and grounding:
				index = passage.find(grounding)
				p1, p2, p3 = passage[:index], passage[index:index+len(grounding)], passage[index+len(grounding):]
				t1, t2, t3 = tokenizer.tokenize(p1), tokenizer.tokenize(p2), tokenizer.tokenize(p3)
				if len(t1) + len(t2) + len(t3) < 300:
					passage = tokenizer.convert_tokens_to_string(t1) + ' <extra_id_0> ' + tokenizer.convert_tokens_to_string(t2) + ' <extra_id_1> ' + tokenizer.convert_tokens_to_string(t3)
				else:
					temp = int((300 - len(t2)) / 2)
					passage = tokenizer.convert_tokens_to_string(t1[len(t1)-temp:]) + ' <extra_id_0> ' + tokenizer.convert_tokens_to_string(t2) + ' <extra_id_1> ' + tokenizer.convert_tokens_to_string(t3[:temp])
				d += 1
			else:
				c += 1
			ctx['text'] = passage
print(c)
print(d)

with open('./data/grounding/val_50_sp_window.json', 'w') as f:
        json.dump(val_data, f, indent=4)
