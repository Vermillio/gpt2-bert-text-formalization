from collections import defaultdict
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

def clean(sentences):
    result = []
    cases = defaultdict(int)
    for ln, line in enumerate(sentences):
    	if line[0] == '=':
            continue
    	lines_cnt += 1

    	sents = sent_tokenize(line)
    	sentences_cnt += len(sents)
    	print(file=outfh)
    	for sent in sents:
    		words = word_tokenize(sent)
    		tmp = sent.replace('#',' ').replace('...','#').replace('!','#').replace('?','#').replace(',','#').replace('--','#').replace('-','#').replace(';','#')\
    			      .replace(':','#').replace('`','#').replace('"','#').replace('.','#').replace(' ','')

    		no_punct = []
    		size = 0
    		for npidx,w in enumerate(words):
    			if w in ('...','!','?',',','--','-',';',':','`','"','.'):
    				no_punct.append(size)
    				size = 0
    			else:
    				size += 1
    		no_punct.append(size)

    		pos = pos_tag(words)
    		skip = False
    		for idx,(w,p) in enumerate(pos[:-1]):
    			if p in ('VB','VBD','VBG','VBN','VBP','VBZ') and pos[idx+1][1] in ('VB','VBD','VBG','VBN','VBP','VBZ'):
    				if p == 'VBD' and pos[idx+1][1] == 'VBN': continue
    				if p == 'VB' and pos[idx+1][1] == 'VBN': continue
    				if p == 'VBP' and pos[idx+1][1] == 'VBN': continue
    				if p == 'VBZ' and pos[idx+1][1] == 'VBN': continue
    				if w == 'been' and pos[idx+1][1] == 'VBN': continue
    				if w in ('be','was','are','is',"'re","'s","been","have") and pos[idx+1][1] == 'VBG': continue
    				if w == 'i': continue
    				cases['verbverb'] += 1
    				skip = True
    				break
    		if set(words)-vocab:
    			cases['new_word'] += 1
    			skip = True
    		elif max(no_punct)>25:
    			cases['no_punctuation'] += 1
    			skip = True
    		elif len(words)>=60:
    			cases['too_long'] += 1
    			skip = True
    		elif "###" in tmp:
    			cases['too_much_punctuation'] += 1
    			skip = True
    		for idx,w in enumerate(words[:-1]):
    			if w == words[idx+1]:
    				cases['duplicate_words'] += 1
    				skip = True
    				break
    		if sent[-1] not in '.!?':
    			cases['bad_end'] += 1
    			skip = True

    		if not skip:
    			result.append(sent)

    return result, cases
