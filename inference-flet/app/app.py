# "msizes 8/7 disagree" error in VS code terminal
# Sol: https://github.com/dagster-io/dagster/discussions/12721
 
import vertexai
from vertexai.language_models import TextGenerationModel


import flet as ft
PROJECT_ID = "argolis-rafaelsanchez-ml-dev"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)


import vertexai
from vertexai.language_models import TextGenerationModel

parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@001")

def main(page: ft.Page):

    # def button_clicked(e):
    #     print(f"Textboxes values are:  '{slider_temp.value}', '{slider_tokens.value}', '{slider_topp.value}', '{slider_topk.value}'.")
    #     answer = model.predict(
    #         f'{prompt.value}',
    #         max_output_tokens=int(f'{slider_tokens.value}'),#{slider_tokens.value}, # default 128
    #         temperature=float(f'{slider_temp.value}'), # default 0
    #         top_p=float(f'{slider_topp.value}'),#{slider_topp.value}, # default 1
    #         top_k=int(f'{slider_topk.value}'))#{slider_topk.value}) # default 40
    #     print(answer)
    #     result.value = answer
    #     page.update()  


    result = ft.Text("Result")#, text_align=TextAllign.LEFT)

    # https://gallery.flet.dev/icons-browser/
    prompt = ft.TextField(label="Enter prompt", hint_text="Please enter prompt here", icon=ft.icons.GENERATING_TOKENS)

    temp = ft.Text(value="Temperature (0-1)")
    slider_temp = ft.Slider(min=0, max=1, divisions=0.1, value = 0.2, label="{value}%")

    tokens = ft.Text(value="Output tokens (1-8192)")
    slider_tokens = ft.Slider(min=1, max=8192, divisions=256, value = 1024, label="{value}")

    topp = ft.Text(value="Top P (0-1)")
    slider_topp =ft.Slider(min=0, max=1, divisions=0.1, value = 0.9, label="{value}%")

    topk = ft.Text(value="Top K (0-40)")
    slider_topk = ft.Slider(min=0, max=40, divisions=1, value = 40, label="{value}%")
    
    b = ft.ElevatedButton(text="Submit", on_click=button_clicked)

    left_col = ft.Column([prompt, temp, slider_temp, tokens, slider_tokens, topp, slider_topp, topk, slider_topk, b])
    right_col = ft.Column([ result ])

    page.add(ft.Row([left_col, right_col], spacing=10))


    #page.add(result)
    #page.add(prompt, temp, slider_temp, tokens, slider_tokens, topp, slider_topp, topk, slider_topk, b, t)

ft.app(target=main, port=7080)


def main(page: ft.Page):
    page.add(ft.Text(value="Hello, world!"))

ft.app(target=main)
