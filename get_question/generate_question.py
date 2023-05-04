import openai
import openai_config
from docx import Document

openai.api_key = openai_config.openai_key

base_prompt = '作为一个老师，你需要根据以下文本给出尽可能符合文本意思的JSON格式问答题，以评估学生对此段文本的理解能力。要求：\n' \
              '1. 问题不能超出所给文本的范围。\n' \
              '2. 问题的长度不超过20个字，尽量简短地阐述问题。\n' \
              '3. 答案必须能够准确且详实地回答问题且要符合文本原意，每个问题的答案不能少于10个字，但不能超过50个字。\n' \
              '4. 所有问答题必须使用中文表述。\n' \
              '5. 如果遇到无法处理的指令（只靠文本无法回答），给出无法处理的回复。\n' \
              '6. 每个问答题对应的JSON格式是：{{"instruction":问题,"input":"",output:答案}}\n' \
              '需要你生成JSON格式问答题的文本如下所示：\n{}\n' \
              '现在给出{}条JSON格式问答题，注意，以JSON数组格式输出：\n'


def get_prompts(doc_path='../source_data/1.docx', max_text_len=1300):
    prompts = []
    doc = Document(doc_path)
    text = ''
    p = doc.paragraphs
    i = 0
    while i < len(p):
        if len(text) + len(p[i].text) > max_text_len:
            if len(p[i].text) == 0:
                i += 1
                continue
            text = text.replace('\n', '').replace(' ', '').replace('A', '').replace('★', '').strip()
            prompt = base_prompt.format(text, len(text) // 100)
            prompts.append(prompt)
            text = ''
        else:
            text += p[i].text
            i += 1
    return prompts


def get_questions(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=2048,
        top_p=0.7,
        frequency_penalty=1.02,
        presence_penalty=2
    )
    return response


if __name__ == '__main__':
    prompt_list = get_prompts('../source_data/2.docx')
    with open('prompts.txt', 'a', encoding='utf-8') as f:
        for p in prompt_list:
            f.write(p)
            f.write('\n')
