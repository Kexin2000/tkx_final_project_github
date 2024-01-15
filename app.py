import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
import requests
from cachetools import cached, TTLCache


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i + word_length]
            if (i + word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx + 1)):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx + start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:

    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(openAI_key, prompt, model="gpt-3.5-turbo"):
    openai.api_key = openAI_key
    temperature = 0.7
    max_tokens = 256
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    if model == "text-davinci-003":
        completions = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        message = completions.choices[0].text
    else:
        message = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "Here is some initial assistant message."},
                {"role": "user", "content": prompt}
            ],
            temperature=.3,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ).choices[0].message['content']
    return message


def generate_answer(question, openAI_key, model):
    topn_chunks = recommender(question)
    prompt = 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "Cite each reference using [ Page Number] notation. " \
              "Only answer what is asked. The answer should be short and concise. \n\nQuery: "

    prompt += f"{question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, model)
    return answer


def question_answer(chat_history, url, file, question, openAI_key, model):
    try:
        if openAI_key.strip() == '':
            return '[ERROR]: Please enter your Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'
        if url.strip() == '' and file is None:
            return '[ERROR]: Both URL and PDF is empty. Provide at least one.'
        if url.strip() != '' and file is not None:
            return '[ERROR]: Both URL and PDF is provided. Please provide only one (either URL or PDF).'
        if model is None or model == '':
            return '[ERROR]: You have not selected any model. Please choose an LLM model.'
        if url.strip() != '':
            glob_url = url
            download_pdf(glob_url, 'corpus.pdf')
            load_recommender('corpus.pdf')
        else:
            old_file_name = file.name
            file_name = file.name
            file_name = file_name[:-12] + file_name[-4:]
            os.rename(old_file_name, file_name)
            load_recommender(file_name)
        if question.strip() == '':
            return '[ERROR]: Question field is empty'
        if model == "text-davinci-003" or model == "gpt-4" or model == "gpt-4-32k":
            answer = generate_answer_text_davinci_003(question, openAI_key)
        else:
            answer = generate_answer(question, openAI_key, model)
        chat_history.append([question, answer])
        return chat_history
    except openai.error.InvalidRequestError as e:
        return f'[ERROR]: Either you do not have access to GPT4 or you have exhausted your quota!'


def generate_text_text_davinci_003(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer_text_davinci_003(question, openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "Cite each reference using [ Page Number] notation (every result has this number at the beginning). " \
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects " \
              "with the same name, create separate answers for each. Only include information found in the results and " \
              "don't add any additional information. Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier " \
              "search results which has nothing to do with the question. Only answer what is asked. The " \
              "answer should be short and concise. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text_text_davinci_003(openAI_key, prompt, "text-davinci-003")
    return answer


# pre-defined questions
questions = ["这项研究调查了什么？",    
             "你能提供这篇论文的摘要吗？",    
             "这项研究使用了哪些方法论？",    
             "这项研究使用了哪些数据间隔？请告诉我开始日期和结束日期？",    
             "这项研究的主要局限性是什么？",    
             "这项研究的主要缺点是什么？",    
             "这项研究的主要发现是什么？",    
             "这项研究的主要结果是什么？",    
             "这项研究的主要贡献是什么？",    
             "这篇论文的结论是什么？",    
             "这项研究中使用了哪些输入特征？",    
             "这项研究中的因变量是什么？",
            ]


# =============================================================================
CACHE_TIME = 60 * 60 * 6  # 6 hours


def parse_arxiv_id_from_paper_url(url):
    return url.split("/")[-1]


@cached(cache=TTLCache(maxsize=500, ttl=CACHE_TIME))
def get_recommendations_from_semantic_scholar(semantic_scholar_id: str):
    try:
        r = requests.post(
            "https://api.semanticscholar.org/recommendations/v1/papers/",
            json={
                "positivePaperIds": [semantic_scholar_id],
            },
            params={"fields": "externalIds,title,year", "limit": 10},
        )
        return r.json()["recommendedPapers"]
    except KeyError as e:
        raise gr.Error(
            "Error getting recommendations, if this is a new paper it may not yet have"
            " been indexed by Semantic Scholar."
        ) from e


def filter_recommendations(recommendations, max_paper_count=5):
    # include only arxiv papers
    arxiv_paper = [
        r for r in recommendations if r["externalIds"].get("ArXiv", None) is not None
    ]
    if len(arxiv_paper) > max_paper_count:
        arxiv_paper = arxiv_paper[:max_paper_count]
    return arxiv_paper


@cached(cache=TTLCache(maxsize=500, ttl=CACHE_TIME))
def get_paper_title_from_arxiv_id(arxiv_id):
    try:
        return requests.get(f"https://huggingface.co/api/papers/{arxiv_id}").json()[
            "title"
        ]
    except Exception as e:
        print(f"Error getting paper title for {arxiv_id}: {e}")
        raise gr.Error("Error getting paper title for {arxiv_id}: {e}") from e


def format_recommendation_into_markdown(arxiv_id, recommendations):
    # title = get_paper_title_from_arxiv_id(arxiv_id)
    # url = f"https://huggingface.co/papers/{arxiv_id}"
    # comment = f"Recommended papers for [{title}]({url})\n\n"
    comment = "The following papers were recommended by the Semantic Scholar API \n\n"
    for r in recommendations:
        hub_paper_url = f"https://huggingface.co/papers/{r['externalIds']['ArXiv']}"
        comment += f"* [{r['title']}]({hub_paper_url}) ({r['year']})\n"
    return comment


def return_recommendations(url):
    arxiv_id = parse_arxiv_id_from_paper_url(url)
    recommendations = get_recommendations_from_semantic_scholar(f"ArXiv:{arxiv_id}")
    filtered_recommendations = filter_recommendations(recommendations)
    return format_recommendation_into_markdown(arxiv_id, filtered_recommendations)

# ==============================================================================================

def run_gradio_app():

    # 第一个文件的内容
    title_1 = "相关文献导航系统"
    description_1 = (
        "将一篇论文的链接粘贴到下方方框处，然后从文献导航系统获取类似论文的推荐。"
        "注意：如果论文是新的或尚未被文献导航系统索引，可能无法推荐。"
    )
    examples_1 = [
        "https://huggingface.co/papers/2309.12307",
        "https://huggingface.co/papers/2211.10086",
    ]

    # 第二个文件的内容
    title_2 = "论文解读系统"
    description_2 = (
        "论文解读系统允许你与你的 PDF 文件进行对话。它使用谷歌的通用句子编码器和深度平均网络（DAN）来提供无幻觉的响应，通过提高 OpenAI 的嵌入质量。"
        "它在方括号中注明页码（[页码]），并显示信息的位置，增加了回应的可信度。"
    )

    with gr.Blocks() as tab1:
        interface = gr.Interface(
            return_recommendations,
            gr.Textbox(lines=1),
            gr.Markdown(),
            examples=examples_1,
            title=title_1,
            description=description_1,
        )

    with gr.Blocks() as tab2:
        gr.Markdown(f'<center><h3>{title_2}</h3></center>')
        gr.Markdown(description_2)
        with gr.Row():
            with gr.Group():
                gr.Markdown(
                    f'<p style="text-align:center">获取你的Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>')
                with gr.Accordion("API Key"):
                    openAI_key = gr.Textbox(
                        label='在这里输入您的API key（老师如果需要测试，可以先用我的key：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）')
                    url = gr.Textbox(label='输入pdf链接   (Example: https://arxiv.org/pdf/1706.03762.pdf )')
                    gr.Markdown("<center><h4>OR<h4></center>")
                    file = gr.File(label='在这里上传您的文件', file_types=['.pdf'])
                question = gr.Textbox(label='输入您的问题')
                gr.Examples(
                    [[q] for q in questions],
                    inputs=[question],
                    label="您可能想问？",
                )
                model = gr.Radio([
                    'gpt-3.5-turbo',
                    'gpt-3.5-turbo-16k',
                    'gpt-3.5-turbo-0613',
                    'gpt-3.5-turbo-16k-0613',
                    'text-davinci-003',
                    'gpt-4',
                    'gpt-4-32k'
                ], label='Select Model')
                btn = gr.Button(value='提交')

            with gr.Group():
                chatbot = gr.Chatbot()

        # Bind the click event of the button to the question_answer function
        btn.click(
            question_answer,
            inputs=[chatbot, url, file, question, openAI_key, model],
            outputs=[chatbot],
        )

    # 将两个界面放入一个 Tab 应用中
    demo = gr.TabbedInterface([tab1, tab2], ["相关文献导航系统", "论文解读系统"])
    demo.launch()

if __name__ == '__main__':
    # 这将确保只有当直接运行此文件时才启动 Gradio 应用
    run_gradio_app()
