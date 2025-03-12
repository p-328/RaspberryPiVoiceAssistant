from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import json
import multiprocessing
import os

# The purpose of this class is to modularize the whole chatbot object and also modularize the training process
class Model:
    _chatbot = ChatBot('Answerer')

    # Initializes model
    def __init__(self):
        questions, answers = self._load_data()
        input('Press key to continue')
        self._train_data(questions, answers)

    # Gets files from data source
    def _get_data_source(self, filename):
        return os.path.join(os.path.join(os.getcwd(), "data"), filename)

    # This method loads Stanford's SQuAD dataset for natural language processing
    def _preprocess_squad(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        questions = []
        answers = []
        # Filtering through JSON Data for answers
        for item in data['data']:
            for paragraph in item['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    if len(qa['answers']) != 0:
                        answer = qa['answers'][0]['text']
                        questions.append(question)
                        answers.append(answer)

        return questions, answers
    
    # Processes list of JSONL JSON Objects in another dataset
    def _preprocess_training_jsonl(self, filename):
        with open(filename) as f:
            data = list(f)
        data = [json.loads(json_str) for json_str in data]
        return data
    
    # Processes data list from self._preprocess_training_jsonl into QA format
    def _load_train_rand_split(self):
        data = self._preprocess_training_jsonl(self._get_data_source("train_rand_split.jsonl"))
        questions = [json_obj['question']['stem'] for json_obj in data]
        answers = []
        for json_obj in data:
            answer_key = json_obj['answerKey']
            for choice in json_obj['question']['choices']:
                if choice['label'] == answer_key:
                    answers.append(choice['text'])
        return questions, answers
    
    # Reads Question Answer TXT Dataset from another source
    def _read_qa_data_txt(self, filename):
        with open(filename) as f:
            data = [line.strip().split('\t') for line in f.readlines()]
        data[0][0] = "ArticleTitle"
        qas = [[phr.strip() for phr in data_row[1:3]] for data_row in data]
        return [row[0] for row in qas], [row[1] for row in qas]
    
    # Loads all of the QA data from all of the data sources in the 'data' directory
    def _load_data(self):
        questions, answers = self._preprocess_squad(self._get_data_source("train-v2.0.json"))
        questions2, answers2 = self._read_qa_data_txt(self._get_data_source("S08_question_answer_pairs.txt"))
        questions3, answers3 = self._read_qa_data_txt(self._get_data_source("S08_question_answer_pairs.txt"))
        questions4, answers4 = self._read_qa_data_txt(self._get_data_source("S08_question_answer_pairs.txt"))
        questions5, answers5 = self._load_train_rand_split()
        questions.extend(
            questions2.extend(
                questions3.extend(
                    questions4.extend(
                        questions5
                    )
                )
            )
        )
        answers.extend(
            answers2.extend(
                answers3.extend(
                    answers4.extend(
                        answers5
                    )
                )
            )
        )
        return questions, answers

    # Trains on a singular question and answer
    def _train(self, qa_tuple):
        q, a = qa_tuple
        self.trainer = ListTrainer(self._chatbot)
        trainer.train([q, a])
    
    # Trains the chatbot off of the QA data. Uses multiprocessing to yield more efficiency.
    def _train_data(self, questions, answers):
        trainer = ListTrainer(self._chatbot)
        data = [(q, a) for q, a in zip(questions, answers)]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            p.map(self._train, data)

    # Answers question using chatbot    
    def answer(self, question):
        return self._chatbot.get_response(question)
