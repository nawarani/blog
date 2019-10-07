Title: Intro to Naural Language Processing with spaCy
Date: 10-07-2019
Slug: blog-3

## spaCy

spaCy is a natural language processing library for python. Some of its features are:
- non-destructive tokenization,
- named entity recognition,
- pre-trained models
- support for 52+ individual langueages.

Since coding is best explained through examples, in this blog, we will use spacy to work through a book from [Project Gutenberg](http://www.gutenberg.org/ebooks/1661).

### Finding proper nouns and custom patterns in The Adventures of Sherlock Holmes

#### Install spacy and language model


```python
! pip install -U spacy
! python -m spacy download en_core_web_sm
```

#### Imports


```python
import spacy
```

#### Load model as the nlp onject

spacy language and models can be downloaded as python packages. Some of the models and langugae packages can be found [here](https://spacy.io/usage/models). For this example, we will be using the small english languge web model 'en_core_web_sm'.


```python
nlp = spacy.load('en_core_web_sm')
```

#### Load text


```python
text = open('data/1661-0.txt').read()
```

The length of the text is more than the default max length, so we reassign the max length of text to reflect that.


```python
nlp.max_length = len(text)
```

#### Process text


```python
doc = nlp(text)
```

#### Find sentences in text


```python
sentences = list(doc.sents)
```

we can access the sentences through a loop.


```python
long_sentences = [sentence for sentence in sentences if len(sentence) > 100]
long_sentences[0]
```




    A shock of orange hair, a
    pale face disfigured by a horrible scar, which, by its contraction, has
    turned up the outer edge of his upper lip, a bulldog chin, and a pair
    of very penetrating dark eyes, which present a singular contrast to the
    colour of his hair, all mark him out from amid the common crowd of
    mendicants and so, too, does his wit, for he is ever ready with a reply
    to any piece of chaff which may be thrown at him by the passers-by.



#### Tokenization, Lemmatization and labeling of named entities

When we call the nlp object on our text, it calls for a pipeline object that tokenizes, tags, parses and recognizes entities. We can access the tokens and entities through a loop, call for the lemmas of each word through .lemma_ and find labels for the entities through .label_

![](https://d33wubrfki0l68.cloudfront.net/16b2ccafeefd6d547171afa23f9ac62f159e353d/48b91/pipeline-7a14d4edd18f3edfee8f34393bff2992.svg)


```python
token_list = [token for token in doc]
token_list[1000:1010]
```




    [problem, ., I, rang, the, bell, , and, was, shown]




```python
lemma_list = [token.lemma_ for token in doc]
lemma_list[1000:1010]
```




    ['problem', '.', '-PRON-', 'ring', 'the', 'bell', '\n', 'and', 'be', 'show']




```python
entity_list = []
label_list = []
for ent in doc.ents:
    entity_list.append(ent)
    label_list.append(ent.label_)
i = 0
while i < 10:
    print(entity_list[i], ':', label_list[i])
    i += 1
```

    Gutenberg : PERSON
    The Adventures of Sherlock Holmes : WORK_OF_ART
    Arthur Conan Doyle

     : ORG
    the Project Gutenberg License : ORG
    eBook : LAW
    The Adventures of Sherlock Holmes

     : WORK_OF_ART
    Arthur Conan Doyle

    Release Date : PERSON
    November 29, 2002 : DATE
    May 20, 2019 : DATE
    English : LANGUAGE


As we can see, it's not perfect. Often times, for bigger text documents like this it is better to use larger models. 'en_core_web_sm' is a smaller model, as denoted by the 'sm' suffix.

#### Find all the proper nouns in the text


```python
proper_noun_list = [token.text for token in doc if token.pos_ == 'PROPN']
print('Number of proper nouns in text:', len(proper_noun_list))
print('Example:', proper_noun_list[1537:1550])
```

    Number of proper nouns in text: 5070
    Example: ['Miss', 'Turner', 'James', 'Mr.', 'Holmes', 'Miss', 'Turner', 'God', 'Holmes', 'Lestrade', 'James', 'McCarthy', 'Holmes']


The proper nouns can also be accessed through loops.

#### Stop words

Stop words and punctuations can be targeted through is_stop and is_punct.


```python
clean_doc = [token.text for token in doc if (not token.is_stop) and (not token.is_punct)]
clean_doc[0:10]
```




    ['\n',
     'Project',
     'Gutenberg',
     'Adventures',
     'Sherlock',
     'Holmes',
     'Arthur',
     'Conan',
     'Doyle',
     '\n\n']



We can use our own stop_words or make a default stop_word not a stop_word with in the following [process](https://stackoverflow.com/questions/41170726/add-remove-stop-words-with-spacy).


```python
nlp.vocab["the"].is_stop = False
nlp.vocab["\n"].is_stop = True
clean_text = [token.text for token in doc if (not token.is_stop) and (not token.is_punct)]
clean_text[0:10]
```




    ['Project',
     'Gutenberg',
     'Adventures',
     'Sherlock',
     'Holmes',
     'Arthur',
     'Conan',
     'Doyle',
     '\n\n',
     'eBook']



#### dependencies and visualization through displacy


```python
# Print the text and the predicted part-of-speech tag
import pandas as pd
print('Sentence:',sentences[637])
sent_df = pd.DataFrame(columns = ['text', 'pos', 'dep', 'related word'])
text_list = []
pos_list = []
dep_list = []
rel_list = []
for word in sentences[637]:
    text_list.append(word.text)
    pos_list.append(word.pos_)
    dep_list.append(word.dep_)
    rel_list.append(word.head.text)
sent_df['text'] = text_list
sent_df['pos'] = pos_list
sent_df['dep'] = dep_list
sent_df['related word'] = rel_list
sent_df
```

    Sentence: I wish she had been of my own station!






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>pos</th>
      <th>dep</th>
      <th>related word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I</td>
      <td>PRON</td>
      <td>nsubj</td>
      <td>wish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wish</td>
      <td>VERB</td>
      <td>ROOT</td>
      <td>wish</td>
    </tr>
    <tr>
      <th>2</th>
      <td>she</td>
      <td>PRON</td>
      <td>nsubj</td>
      <td>been</td>
    </tr>
    <tr>
      <th>3</th>
      <td>had</td>
      <td>AUX</td>
      <td>aux</td>
      <td>been</td>
    </tr>
    <tr>
      <th>4</th>
      <td>been</td>
      <td>AUX</td>
      <td>ccomp</td>
      <td>wish</td>
    </tr>
    <tr>
      <th>5</th>
      <td>of</td>
      <td>ADP</td>
      <td>prep</td>
      <td>been</td>
    </tr>
    <tr>
      <th>6</th>
      <td>my</td>
      <td>PRON</td>
      <td>poss</td>
      <td>station</td>
    </tr>
    <tr>
      <th>7</th>
      <td>own</td>
      <td>ADJ</td>
      <td>amod</td>
      <td>station</td>
    </tr>
    <tr>
      <th>8</th>
      <td>station</td>
      <td>NOUN</td>
      <td>pobj</td>
      <td>of</td>
    </tr>
    <tr>
      <th>9</th>
      <td>!</td>
      <td>PUNCT</td>
      <td>punct</td>
      <td>wish</td>
    </tr>
    <tr>
      <th>10</th>
      <td>\n</td>
      <td>SPACE</td>
      <td></td>
      <td>!</td>
    </tr>
  </tbody>
</table>
</div>




```python
from spacy import displacy
displacy.render(sentences[6751], style = 'dep')
```


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="c1d4b15930c0417caaf7354a0c9d9d82-0" class="displacy" width="3900" height="574.5" direction="ltr" style="max-width: none; height: 574.5px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="50">“</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PUNCT</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="225">‘</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">PUNCT</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="400">Jephro,’</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="575">said</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="750">she, ‘</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">PRON</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="925">there</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="925">PRON</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="1100">is</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1100">AUX</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="1275">an</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1275">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="1450">impertinent</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1450">ADJ</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="1625">fellow</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1625">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="1800">upon</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1800">SCONJ</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="1975">the</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="1975">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="2150">road</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2150">NOUN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="2325">
</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2325">SPACE</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="2500">there</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2500">ADV</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="2675">who</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2675">PRON</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="2850">stares</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="2850">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="3025">up</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3025">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="3200">at</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3200">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="3375">Miss</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3375">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="3550">Hunter.’</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3550">PROPN</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="484.5">
    <tspan class="displacy-word" fill="currentColor" x="3725">

</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="3725">SPACE</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-0" stroke-width="2px" d="M70,439.5 C70,264.5 385.0,264.5 385.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">punct</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,441.5 L62,429.5 78,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-1" stroke-width="2px" d="M245,439.5 C245,352.0 380.0,352.0 380.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">punct</textPath>
    </text>
    <path class="displacy-arrowhead" d="M245,441.5 L237,429.5 253,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-2" stroke-width="2px" d="M420,439.5 C420,352.0 555.0,352.0 555.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">dep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M555.0,441.5 L563.0,429.5 547.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-3" stroke-width="2px" d="M595,439.5 C595,352.0 730.0,352.0 730.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M730.0,441.5 L738.0,429.5 722.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-4" stroke-width="2px" d="M945,439.5 C945,352.0 1080.0,352.0 1080.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-4" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">expl</textPath>
    </text>
    <path class="displacy-arrowhead" d="M945,441.5 L937,429.5 953,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-5" stroke-width="2px" d="M595,439.5 C595,177.0 1090.0,177.0 1090.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-5" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">ccomp</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1090.0,441.5 L1098.0,429.5 1082.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-6" stroke-width="2px" d="M1295,439.5 C1295,264.5 1610.0,264.5 1610.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-6" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">det</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1295,441.5 L1287,429.5 1303,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-7" stroke-width="2px" d="M1470,439.5 C1470,352.0 1605.0,352.0 1605.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-7" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">amod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1470,441.5 L1462,429.5 1478,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-8" stroke-width="2px" d="M1120,439.5 C1120,177.0 1615.0,177.0 1615.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-8" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">attr</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1615.0,441.5 L1623.0,429.5 1607.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-9" stroke-width="2px" d="M1645,439.5 C1645,352.0 1780.0,352.0 1780.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-9" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1780.0,441.5 L1788.0,429.5 1772.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-10" stroke-width="2px" d="M1995,439.5 C1995,352.0 2130.0,352.0 2130.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-10" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">det</textPath>
    </text>
    <path class="displacy-arrowhead" d="M1995,441.5 L1987,429.5 2003,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-11" stroke-width="2px" d="M1820,439.5 C1820,264.5 2135.0,264.5 2135.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-11" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2135.0,441.5 L2143.0,429.5 2127.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-12" stroke-width="2px" d="M2170,439.5 C2170,352.0 2305.0,352.0 2305.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-12" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle"></textPath>
    </text>
    <path class="displacy-arrowhead" d="M2305.0,441.5 L2313.0,429.5 2297.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-13" stroke-width="2px" d="M2170,439.5 C2170,264.5 2485.0,264.5 2485.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-13" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">advmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2485.0,441.5 L2493.0,429.5 2477.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-14" stroke-width="2px" d="M2695,439.5 C2695,352.0 2830.0,352.0 2830.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-14" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2695,441.5 L2687,429.5 2703,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-15" stroke-width="2px" d="M2170,439.5 C2170,89.5 2845.0,89.5 2845.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-15" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">relcl</textPath>
    </text>
    <path class="displacy-arrowhead" d="M2845.0,441.5 L2853.0,429.5 2837.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-16" stroke-width="2px" d="M2870,439.5 C2870,352.0 3005.0,352.0 3005.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-16" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prt</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3005.0,441.5 L3013.0,429.5 2997.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-17" stroke-width="2px" d="M2870,439.5 C2870,264.5 3185.0,264.5 3185.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-17" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3185.0,441.5 L3193.0,429.5 3177.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-18" stroke-width="2px" d="M3395,439.5 C3395,352.0 3530.0,352.0 3530.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-18" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">compound</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3395,441.5 L3387,429.5 3403,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-19" stroke-width="2px" d="M420,439.5 C420,2.0 3550.0,2.0 3550.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-19" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">punct</textPath>
    </text>
    <path class="displacy-arrowhead" d="M3550.0,441.5 L3558.0,429.5 3542.0,429.5" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-20" stroke-width="2px" d="M3570,439.5 C3570,352.0 3705.0,352.0 3705.0,439.5" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-c1d4b15930c0417caaf7354a0c9d9d82-0-20" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle"></textPath>
    </text>
    <path class="displacy-arrowhead" d="M3705.0,441.5 L3713.0,429.5 3697.0,429.5" fill="currentColor"/>
</g>
</svg>


While the code block mentioned above is enough to render displacy in jupyter notebook, we need to use displacy.serve for other cases.

#### Matcher

So if we wanted to extract every word group that consisted of a noun following an adjective, how would we do that?

We can use Rule based matching to accomplish this.


```python
sentences[6751]
```




    “‘Jephro,’ said she, ‘there is an impertinent fellow upon the road
    there who stares up at Miss Hunter.’




Import matcher.


```python
from spacy.matcher import Matcher
```

Instantiate the matcher with shared vocab of the doc.


```python
matcher = Matcher(nlp.vocab)
```

Create a custom pattern and add it to the matcher.


```python
pattern = [{'POS': 'ADJ'},
           {'POS': 'NOUN', 'OP': '+'}]
matcher.add('CUSTOM_PATTERN', None, pattern)
```

note: "OP" can have one of four values:
- '!' for 0 times
- '?' for 0 or 1 time
- '+' for 1 or more times
- '*' for 0 or more times

We can add pattern to the matcher as well thorugh matcher.add()


```python
matches = matcher(doc)
len(matches)
```




    3202



We can access the matched spans through loops.


```python
match_list = [doc[start:end] for match_id, start, end in matches]
match_list[100:105]
```




    [other purposes, private note, own seal, little problem, serious one]



There's a lot more we can do with spacy, it is really well [documented](https://spacy.io/usage/spacy-101) and easy to pick up. So if you're looking to do some more NLP alongside what [nltk](https://www.nltk.org/) has to offer, check out spacy!
