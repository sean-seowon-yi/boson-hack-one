# -*- coding: utf-8 -*-
import json
import os
import re
import time
import traceback
from dotenv import load_dotenv
from loguru import logger

# Boson-backed wrappers (already updated in your project)
from tools.step031_translation_openai import openai_response
from tools.step032_translation_llm import llm_response
from tools.step033_translation_translator import translator_response
from tools.step035_translation_qwen import qwen_response

load_dotenv()

def get_necessary_info(info: dict):
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        'tags': info['tags'],
    }

def ensure_transcript_length(transcript, max_length=4000):
    mid = len(transcript) // 2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length // 2
    return before[:length] + after[-length:]

def split_text_into_sentences(para):
    # sentence split tuned for Chinese punctuation
    para = re.sub('([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    return para.rstrip().split("\n")

def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = (result
              .replace('²', '的平方')
              .replace('————', '：')
              .replace('——', '：')
              .replace('°', '度'))
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result

def valid_translation(text, translation):
    # try to extract clean translation from common wrappers
    if translation.startswith('```') and translation.endswith('```'):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)

    if ((translation.startswith('“') and translation.endswith('”')) or
        (translation.startswith('"') and translation.endswith('"'))):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    for pattern in ['：“', '："', ':"', ': "']:
        if any(k in translation for k in ['翻译', '译文', 'Translation', 'translate']):
            if pattern in translation and ('”' in translation or '"' in translation):
                right_quote = '”' if '”' in translation else '"'
                translation = translation.split(pattern)[-1].split(right_quote)[0]
                return True, translation_postprocess(translation)

    # length sanity
    if len(text) <= 10:
        if len(translation) > 15:
            return False, 'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text) * 0.75:
        return False, 'The translation is too long. Only translate the following sentence and give me the result.'

    # forbid meta text
    forbidden = ['翻译', '译文', '这句', '\n', '简体中文', '中文',
                 'translate', 'Translate', 'translation', 'Translation']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation)

def split_sentences(translation, use_char_based_end=True):
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item.get('translation', '')

        if not translation_text or len(translation_text.strip()) == 0:
            output_data.append({
                "start": round(start, 3),
                "end": round(item['end'], 3),
                "text": text,
                "speaker": speaker,
                "translation": translation_text or "未翻译",
            })
            continue

        sentences = split_text_into_sentences(translation_text)
        duration_per_char = ((item['end'] - item['start']) / max(1, len(translation_text))
                             if use_char_based_end else 0)

        for sentence in sentences:
            sentence_end = start + duration_per_char * len(sentence) if use_char_based_end else item['end']
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence
            })
            if use_char_based_end:
                start = sentence_end

    return output_data

def summarize(info, transcript, target_language='简体中文', method='LLM'):
    transcript = ' '.join(line['text'] for line in transcript)
    transcript = ensure_transcript_length(transcript, max_length=2000)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '

    if method in ['Google Translate', 'Bing Translate']:
        full_description = f'{info_message}\n{transcript}\n{info_message}\n'
        translation = translator_response(full_description, target_language)
        return {
            'title': translator_response(info['title'], target_language),
            'author': info['uploader'],
            'summary': translation,
            'language': target_language
        }

    full_description = (
        f'The following is the full content of the video:\n{info_message}\n{transcript}\n{info_message}\n'
        'According to the above content, detailedly Summarize the video in JSON format:\n'
        '```json\n{"title": "", "summary": ""}\n```'
    )

    retry_message = ''
    success = False
    for _ in range(9):
        try:
            messages = [
                {'role': 'system',
                 'content': ('You are an expert in the field of this video. '
                             'Please summarize the video in JSON format.\n'
                             '```json\n{"title": "the title of the video", "summary", "the summary of the video"}\n```')},
                {'role': 'user', 'content': full_description + retry_message},
            ]

            # Map methods to our Boson-backed clients
            if method == 'LLM':
                response = llm_response(messages)               # transformers local Qwen if configured
            elif method == 'OpenAI':
                response = openai_response(messages)             # Boson OpenAI-compatible (Qwen3…)
            elif method in ['阿里云-通义千问', 'Qwen']:
                response = qwen_response(messages)               # Boson Qwen wrapper
            else:
                raise Exception('Invalid method')

            summary = response.replace('\n', '')
            if '视频标题' in summary:
                raise Exception("包含“视频标题”")
            logger.info(summary)
            summary = re.findall(r'\{.*?\}', summary)[0]
            summary = json.loads(summary)
            summary = {
                'title': summary['title'].replace('title:', '').strip(),
                'summary': summary['summary'].replace('summary:', '').strip()
            }
            if not summary['title'] or not summary['summary'] or 'title' in summary['title']:
                raise Exception('Invalid summary')

            success = True
            break
        except Exception as e:
            traceback.print_exc()
            retry_message += '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'
            logger.warning(f'Summary failed\n{e}')
            time.sleep(1)

    if not success:
        raise Exception('Summary failed')

    # Return original-language summary (downstream will handle translation to target_language)
    messages = [
        {'role': 'system',
         'content': (f'You are a native speaker of {target_language}. Please translate the title and summary into '
                     f'{target_language} in JSON format. ```json\n'
                     f'{{"title": "the {target_language} title of the video", '
                     f'"summary", "the {target_language} summary of the video", '
                     f'"tags": [list of tags in {target_language}]}}\n```.')},
        {'role': 'user',
         'content': (f'The title of the video is "{summary["title"]}". The summary of the video is "{summary["summary"]}". '
                     f'Tags: {info["tags"]}.\nPlease translate the above title and summary and tags into {target_language} '
                     f'in JSON format. ```json\n{{"title": "", "summary", ""， "tags": []}}\n```. '
                     f'Remember to translate the title and the summary and tags into {target_language} in JSON.')},
    ]
    while True:
        try:
            logger.info(summary)
            if target_language in summary['title'] or target_language in summary['summary']:
                raise Exception('Invalid translation')
            title = summary['title'].strip()
            if ((title.startswith('"') and title.endswith('"')) or
                (title.startswith('“') and title.endswith('”')) or
                (title.startswith('‘') and title.endswith('’')) or
                (title.startswith("'") and title.endswith("'")) or
                (title.startswith('《') and title.endswith('》'))):
                title = title[1:-1]
            result = {
                'title': title,
                'author': info['uploader'],
                'summary': summary['summary'],
                'tags': info['tags'],
                'language': target_language
            }
            return result
        except Exception as e:
            logger.warning(f'Title/Summary translation failed\n{e}')
            time.sleep(1)

def _translate(summary, transcript, target_language='简体中文', method='LLM'):
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    full_translation = []

    if target_language == '简体中文':
        fixed_message = [
            {'role': 'system',
             'content': (f'You are an expert in the field of this video.\n{info}\n'
                         f'Translate the sentence into {target_language}。下面我让你来充当翻译家，你的目标是把任何语言翻译成{target_language}，'
                         f'请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。请将人工智能的“agent”翻译为“智能体”，'
                         f'强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译正确和简洁。注意信达雅。')},
            {'role': 'user', 'content': f'使用地道的{target_language}Translate:"Knowledge is power."'},
            {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
            {'role': 'user', 'content': f'使用地道的{target_language}Translate:"To be or not to be, that is the question."'},
            {'role': 'assistant', 'content': '翻译：“生存还是毁灭，这是一个值得考虑的问题。”'},
        ]
    else:
        fixed_message = [
            {'role': 'system',
             'content': (f'You are a language expert. Translate the transcript of a video titled "{summary["title"]}". '
                         f'Summary: {summary["summary"]}. Translate each sentence into {target_language} accurately and fluently.')},
            {'role': 'user', 'content': 'Please translate the following text: "Original Text"'},
            {'role': 'assistant', 'content': 'Translated text: "Translated Text"'},
            {'role': 'user', 'content': 'Translate the following text: "Another Original Text"'},
            {'role': 'assistant', 'content': 'Translated text: "Another Translated Text"'},
        ]

    history = []
    for line in transcript:
        text = line['text']
        retry_message = 'Only translate the quoted sentence and give me the final translation.'
        if method == 'Google Translate':
            translation = translator_response(text, to_language=target_language, translator_server='google')
        elif method == 'Bing Translate':
            translation = translator_response(text, to_language=target_language, translator_server='bing')
        else:
            for _ in range(10):
                messages = fixed_message + history[-30:] + [{'role': 'user', 'content': f'Translate:"{text}"'}]
                try:
                    if method == 'LLM':
                        response = llm_response(messages)
                    elif method == 'OpenAI':
                        response = openai_response(messages)
                    elif method in ['阿里云-通义千问', 'Qwen']:
                        response = qwen_response(messages)
                    else:
                        raise Exception('Invalid method')

                    translation = response.replace('\n', '')
                    logger.info(f'Original: {text}')
                    logger.info(f'Translation: {translation}')
                    success, translation = valid_translation(text, translation)
                    if not success:
                        retry_message += translation
                        raise Exception('Invalid translation')
                    break
                except Exception as e:
                    logger.error(e)
                    logger.warning('Translation failed')
                    time.sleep(1)

        full_translation.append(translation)
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': f'翻译：“{translation}”'})
        time.sleep(0.1)

    return full_translation

def translate(method, folder, target_language='简体中文'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True

    info_path = os.path.join(folder, 'download.info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        info = get_necessary_info(info)
    else:
        info = {
            'title': os.path.basename(folder),
            'uploader': 'Unknown',
            'description': 'Unknown',
            'upload_date': 'Unknown',
            'tags': []
        }

    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path, 'r', encoding='utf-8'))
    else:
        summary = summarize(info, transcript, target_language, method)
        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    translation = _translate(summary, transcript, target_language, method)
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    transcript = split_sentences(transcript)
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return summary, transcript

def translate_all_transcript_under_folder(folder, method, target_language):
    summary_json, translate_json = None, None
    for root, _, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            summary_json, translate_json = translate(method, root, target_language)
        elif 'translation.json' in files:
            summary_json = json.load(open(os.path.join(root, 'summary.json'), 'r', encoding='utf-8'))
            translate_json = json.load(open(os.path.join(root, 'translation.json'), 'r', encoding='utf-8'))
    print(summary_json, translate_json)
    return f'Translated all videos under {folder}', summary_json, translate_json

if __name__ == '__main__':
    # Example
    translate_all_transcript_under_folder(r'videos', 'LLM', '简体中文')
