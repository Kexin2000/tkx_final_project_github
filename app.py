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

CACHE_TIME = 60 * 60 * 6  # 6小时

# 全局的推荐器对象
recommender = None

# 第二个功能的全局变量
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
            "获取推荐时出错，如果这是一篇新论文或尚未被Semantic Scholar索引，则可能尚未有推荐。"
        ) from e


def filter_recommendations(recommendations, max_paper_count=5):
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
        print(f"获取论文标题时出错 {arxiv_id}: {e}")
        raise gr.Error(f"获取论文标题时出错 {arxiv_id}: {e}") from e


def format_recommendation_into_markdown(arxiv_id, recommendations):
    comment = "以下论文由Semantic Scholar API推荐\n\n"
    for r in recommendations:
        hub_paper_url = f"https://huggingface.co/papers/{r['externalIds']['ArXiv']}"
        comment += f"* [{r['title']}]({hub_paper_url}) ({r['year']})\n"
    return comment


def return_recommendations(url):
    arxiv_id = parse_arxiv_id_from_paper_url(url)
    recommendations = get_recommendations_from_semantic_scholar(f"ArXiv:{arxiv_id}")
    filtered_recommendations = filter_recommendations(recommendations)
    return format_recommendation_into_markdown(arxiv_id, filtered_recommendations)


# Gradio界面
title = 'PDF GPT Turbo'
description = """ PDF GPT Turbo允许您与PDF文件交流。它使用Google的Universal Sentence Encoder与Deep averaging network（DAN）来提供无幻觉的响应，通过提高OpenAI的嵌入质量。它在方括号（[Page No.]）中引用页码，显示信息的位置，增强了响应的可信度。"""

# 预定义的问题
questions = [
    "研究调查了什么？",
    "能否提供本文的摘要？",
    "这项研究使用了什么方法？",
    # 需要时添加更多的问题
]

with gr.Blocks(css="""#chatbot { font-size: 14px; min-height: 1200; }""") as demo:
    gr.Markdown(f'<center><h3>{title}</h3></center>')
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            gr.Markdown(f'<p style="text-align:center">在这里获取您的Open AI API密钥 <a href="https://platform.openai.com/account/api-keys">here</a></p>')
            with gr.Accordion("API Key"):
                openAI_key = gr.Textbox(label='在此输入您的OpenAI API密钥')
                url = gr.Textbox(label='在此输入PDF的URL   (示例: https://arxiv.org/pdf/1706.03762.pdf )')
                gr.Markdown("<center><h4>或<h4></center>")
                file = gr.File(label='在此上传您的PDF/研究论文/书籍', file_types=['.pdf'])
            question = gr.Textbox(label='在此输入您的问题')
            gr.Examples(
                [[q] for q in questions],
                inputs=[question],
                label="预定义问题：点击问题以自动填充输入框，然后按Enter键！",
            )
            model = gr.Radio([
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k',
                'gpt-3.5-turbo-0613',
                'gpt-3.5-turbo-16k-0613',
                'text-davinci-003',
                'gpt-4',
                'gpt-4-32k'
            ], label='选择模型', default='gpt-3.5-turbo')
            btn = gr.Button(value='提交')
            btn.style(full_width=True)
        with gr.Group():
            chatbot = gr.Chatbot(placeholder="聊天历史", label="聊天历史", lines=50, elem_id="chatbot")

    # 将按钮的点击事件绑定到question_answer函数
    btn.click(
        question_answer,
        inputs=[chatbot, url, file, question, openAI_key, model],
        outputs=[chatbot],
    )

    # 第二个标签
    gr.Tab("论文推荐", [
        gr.Textbox(label="输入Hugging Face Papers的URL", lines=1),
        gr.Button("获取推荐", return_recommendations),
        gr.Markdown(),
    ])

demo.launch()
