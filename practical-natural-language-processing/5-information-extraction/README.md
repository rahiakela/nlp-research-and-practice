# Information Extraction

## 🔖 Outline

Information extraction (IE) refers to the NLP task of extracting relevant information from text documents.
An example of IE put to use in real-world applications are the short blurbs we see to the right when we search for a popular figure’s name on Google.
IE is a term that’s used to refer to a range of different tasks of varying complexity. The
overarching goal of IE is to extract “knowledge” from text, and each of these tasks provides different information to do that.

Identifying that the article is about “buyback” or “stock price” relates to the IE task of
keyword or keyphrase extraction (KPE). Identifying Apple as an organization and Luca
Maestri as a person comes under the IE task of named entity recognition (NER). Recognizing
that Apple is not a fruit, but a company, and that it refers to Apple, Inc. and
not some other company with the word “apple” in its name is the IE task of named
entity disambiguation and linking. Extracting the information that Luca Maestri is the
finance chief of Apple refers to the IE task of relation extraction.



## 🗒️ Notebooks

Set of notebooks associated with the chapter. 

1. **[Key Phrase Extraction](https://github.com/rahiakela/practical-natural-language-processing/blob/master/5-information-extraction/1_keyphrase_extraction.ipynb)**: Here we demonstrate Key Phrase Extraction using textacy.

2. **[NER using CRF](https://github.com/rahiakela/practical-natural-language-processing/blob/master/5-information-extraction/2_named_entity_recognition.ipynb)**: Here we demonstrate how to perform NER using Conditional Random Fields.

3. **[NER on CONLL using BERT](https://github.com/rahiakela/practical-natural-language-processing/blob/master/5-information-extraction/3_named_entity_recognition_using_bert.ipynb)**: Here we demonstrate NER on the CONLL dataset using pytorch pre-trained BERT from the HuggingFace Transformers library. 

4. **[NER on CONLL using spaCy](https://github.com/rahiakela/practical-natural-language-processing/blob/master/5-information-extraction/4_named_entity_recognition_using_spacy.ipynb)**: Here we demonstrate NER on the CONLL-NER dataset using spaCy.

5. **[NER issues in spaCy](https://github.com/rahiakela/practical-natural-language-processing/blob/master/5-information-extraction/5_named_entity_recognition_issues.ipynb)**: Here we illustrate some of the issues with the existing NER model in spacy and its sensitivity to text structure.

6. **[Entity Linking](https://github.com/rahiakela/practical-natural-language-processing/blob/master/5-information-extraction/6_named_entity_disambiguation_and_linking.ipynb)**: Here we demonstrate how to extract Named Entity linking information using Azure Text Analytics API.

7. **[Relation Extraction](https://github.com/practical-nlp/practical-nlp/blob/master/Ch5/07_REWatson.ipynb)**: Here we demonstrate Relation Extraction with IBM Watson.

8. **[Duckling](https://github.com/practical-nlp/practical-nlp/blob/master/Ch5/08_Duckling.ipynb)**: Here we demonstrate Temporal IE using duckling.

