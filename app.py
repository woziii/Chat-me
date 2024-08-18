# -*- coding: utf-8 -*-
import gradio as gr
import spaces
import torch
import re
import json
import os
import time
import threading
import multiprocessing
from threading import Thread
from typing import Iterator
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
model_name = "Woziii/llama-3-8b-chat-me"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_MAX_NEW_TOKENS = 250
DEFAULT_MAX_NEW_TOKENS = 70
MAX_INPUT_TOKEN_LENGTH = 2048


LUCAS_KNOWLEDGE_BASE = """

# Informations sur ton concepteur :
Lucas Maurici né à Toulouse, âgé de 24 ans. Je vis actuellement sur Paris mais je viens de Toulouse. J'ai un chien, il s'appelle Archi c'est un pinscher moyen.
### Formation académique : du droit à l'intelligence artificielle
Mon voyage intellectuel a débuté à l'Université de Toulouse, où j'ai obtenu une Licence en droit. Assoiffé de connaissances, j'ai poursuivi avec un Master I en droit du numérique et tiers de confiance à l'Université de La Rochelle. Ma curiosité pour les nouvelles technologies m'a ensuite guidé vers un Master II en droit du numérique spécialisé en intelligence artificielle, de retour à Toulouse. Passionné par l'apprentissage autodidacte, je me forme continuellement. Actuellement, je plonge dans les arcanes du traitement du langage naturel et de l'apprentissage profond grâce à une formation en ligne de Stanford.
### Expériences professionnelles : 
Mon parcours professionnel est aussi varié qu'enrichissant. Depuis 2019, je suis conseiller municipal délégué dans la charmante commune d'Escalquens. J'ai également eu l'opportunité de travailler au ministère de l'Économie et des Finances, où j'ai œuvré pour la protection des données. Mon apprentissage à la préfecture de police de Paris m'a permis d'évoluer du rôle de juriste à celui d'assistant du chef de bureau des associations de sécurité civile. Aujourd'hui, je suis fier de contribuer à l'organisation des Jeux Olympiques de Paris 2024 en tant que conseiller juridique.
### Ambitions et personnalité : un esprit curieux et innovant
Mes compétences juridiques sont complétées par une forte appétence pour la technologie. Autonome et force de proposition, j'aime partager mes idées et collaborer avec mes collègues. Ma curiosité insatiable et mon imagination débordante sont les moteurs de mon développement personnel et professionnel.
### Loisirs et racines : 
Bien que le sport ne soit pas ma priorité, j'ai pratiqué le taekwondo pendant plus d'une décennie durant mon enfance. Toulousain d'adoption, je suis un fervent amateur de rugby. Mes racines sont ancrées dans le pittoresque village de La Franqui, près de Narbonne, où j'ai grandi bercé par la Méditerranée. Et oui, je dis "chocolatine" !
### Passion pour l'IA : explorer les frontières du possible
Actuellement, je consacre une grande partie de mon temps libre à l'exploration des modèles de traitement du langage naturel. Je suis reconnaissant envers des pionniers comme Yann LeCun pour leur promotion de l'open source, qui m'a permis de décortiquer de nombreux modèles d'IA. Mon analyse approfondie d'Albert, l'IA du gouvernement, illustre ma soif de comprendre ces technologies fascinantes.
### Compétences techniques : un mélange unique de créativité et de connaissances
Bien que je ne sois pas un codeur Python chevronné, je comprends sa structure et sais communiquer efficacement avec la machine. Je maîtrise les formats JSON, CSV et XML, et je crée mes propres bases de données d'entraînement. Je suis à l'aise avec les outils de lecture de modèles de langage locaux et les plateformes comme Kaggle, Hugging Face et GitHub.
### Langue et communication : en constante amélioration
Mon anglais, bien que solide en compréhension, est en cours d'amélioration à l'oral. Je l'utilise quotidiennement pour mes recherches en IA, conscient de son importance cruciale dans ce domaine en constante évolution.
### Convictions personnelles et vision sur l'IA : l'humain au cœur de la technologie
Je crois fermement en l'autodidaxie et considère la capacité à communiquer avec les machines comme une compétence essentielle. Pour moi, l'art du prompt est une forme d'expression artistique à part entière. Je suis convaincu que la technologie et l'IA doivent rester des outils au service de l'humain, sans jamais le remplacer ou le rendre dépendant.
### Projets :
Utilisant le Large Langage Model d'Anthropic, BraIAn est un correspondant virtuel conçu pour améliorer votre anglais écrit en vous corrigeant pendant que vous discutez, sans interrompre la conversation. L'idée ? Perfectionner votre anglais de manière naturelle, en discutant tout simplement… 💬
BraIAn est là pour discuter, sans vous juger ni chercher à en savoir plus sur vous. Vous pouvez lui dire ce que vous voulez et être qui vous voulez. 🙌
Pourquoi j'ai créé braIAn : J'ai conçu BraIAn pour aider l'utilisateur à reprendre confiance en lui. Il corrige votre anglais sans interrompre votre conversation et cherche constamment à l'alimenter. Ainsi, l'utilisateur travaille et améliore son anglais tout en discutant de ce qu’il souhaite. Cette idée je l'ai eu, car, durant ma scolarité, j'ai eu beaucoup de mal avec la méthode scolaire.
Pour moi, une bonne IA éducative ne doit pas chercher à enseigner. Cette tâche nécessite des qualités humaines telles que l'empathie ou l'imagination. En revanche l'IA peut aider l'utilisateur à trouver sa méthode d'apprentissage. Elle doit être considérée comme un vivier d'idées et d'informations mis à disposition de l'humain. En créant braIAn, j'ai cherché à reproduire cette philosophie. Une IA qui ne fait pas apprendre l'anglais mais une IA qui discute avec l'utilisateur et qui, discrètement, apporte une correction sans détériorer ce qui compte vraiment : ne pas avoir peur d'essayer et converser.

Contacter Lucas :
- Téléphone 📱 : 0659965152
- Mail ✉️ : maurici.lucas@proton.me
- linkedin de Lucas : https://www.linkedin.com/in/lucas-maurici-8a6a311a5/
- Lien portfolio de Lucas: www.lucasmaurici.com
- Page huggingface 🤗 de Lucas: https://huggingface.co/Woziii
"""

is_first_interaction = True

  
def determine_response_type(message):
# Liste améliorée de mots-clés pour les réponses courtes
    short_response_keywords = [
        "salut", "Salut", "SALUT",
        "bonjour", "Bonjour", "BONJOUR",
        "ça va", "ca va", "Ça va", "Ca va", "ÇA VA", "CA VA",
        "comment tu vas", "Comment tu vas", "COMMENT TU VAS",
        "comment vas tu", "Comment vas tu", "COMMENT VAS TU",
        "comment vas-tu", "Comment vas-tu", "COMMENT VAS-TU",
        "quoi de neuf", "Quoi de neuf", "QUOI DE NEUF",
        "coucou", "Coucou", "COUCOU",
        "hello", "Hello", "HELLO",
        "hi", "Hi", "HI",
        "tu fais quoi", "Tu fais quoi", "TU FAIS QUOI",
        "?!", "?!?", "!?",
        "bye", "Bye", "BYE",
        "au revoir", "Au revoir", "AU REVOIR",
        "à plus", "À plus", "A plus", "a plus", "À PLUS", "A PLUS",
        "bonsoir", "Bonsoir", "BONSOIR",
        "merci", "Merci", "MERCI",
        "d'accord", "D'accord", "D'ACCORD",
        "ok", "Ok", "OK",
        "super", "Super", "SUPER",
        "cool", "Cool", "COOL",
        "génial", "Génial", "GENIAL",
        "wow", "Wow", "WOW", "et toi ", "ET TOI", "Et toi"
    ]

    # Liste améliorée de mots-clés pour les réponses longues
    long_response_keywords = [
        "présente", "Présente", "PRÉSENTE", "presente", "Presente", "PRESENTE",
        "parle moi de", "Parle moi de", "PARLE MOI DE",
        "parle-moi de", "Parle-moi de", "PARLE-MOI DE",
        "explique", "Explique", "EXPLIQUE",
        "raconte", "Raconte", "RACONTE",
        "décris", "Décris", "DÉCRIS", "decris", "Decris", "DECRIS",
        "dis moi", "Dis moi", "DIS MOI",
        "dis-moi", "Dis-moi", "DIS-MOI",
        "détaille", "Détaille", "DÉTAILLE", "detaille", "Detaille", "DETAILLE",
        "précise", "Précise", "PRÉCISE", "precise", "Precise", "PRECISE",
        "vision", "Vision", "VISION",
        "t'es qui", "T'es qui", "T'ES QUI",
        "tu es qui", "Tu es qui", "TU ES QUI",
        "t es qui", "T es qui", "T ES QUI",
        "pourquoi", "Pourquoi", "POURQUOI",
        "comment", "Comment", "COMMENT",
        "quel est", "Quel est", "QUEL EST",
        "quelle est", "Quelle est", "QUELLE EST",
        "peux-tu développer", "Peux-tu développer", "PEUX-TU DÉVELOPPER",
        "peux tu developper", "Peux tu developper", "PEUX TU DEVELOPPER",
        "en quoi consiste", "En quoi consiste", "EN QUOI CONSISTE",
        "qu'est-ce que", "Qu'est-ce que", "QU'EST-CE QUE",
        "que penses-tu de", "Que penses-tu de", "QUE PENSES-TU DE",
        "analyse", "Analyse", "ANALYSE",
        "compare", "Compare", "COMPARE",
        "élabore sur", "Élabore sur", "ÉLABORE SUR", "elabore sur", "Elabore sur", "ELABORE SUR",
        "expérience", "Expérience", "EXPÉRIENCE", "experience", "Experience", "EXPERIENCE",
        "expérience pro", "Expérience pro", "EXPÉRIENCE PRO",
        "experience pro", "Experience pro", "EXPERIENCE PRO",
        "expérience professionnelle", "Expérience professionnelle", "EXPÉRIENCE PROFESSIONNELLE",
        "experience professionnelle", "Experience professionnelle", "EXPERIENCE PROFESSIONNELLE",
        "parcours", "Parcours", "PARCOURS",
        "formation", "Formation", "FORMATION",
        "études", "Études", "ÉTUDES", "etudes", "Etudes", "ETUDES",
        "compétences", "Compétences", "COMPÉTENCES", "competences", "Competences", "COMPETENCES",
        "projets", "Projets", "PROJETS",
        "réalisations", "Réalisations", "RÉALISATIONS", "realisations", "Realisations", "REALISATIONS"
    ]
    message_lower = message.lower()
    
    # Compteurs pour les mots-clés courts et longs
    short_count = sum(keyword.lower() in message_lower for keyword in short_response_keywords)
    long_count = sum(keyword.lower() in message_lower for keyword in long_response_keywords)
    
    # Logique de décision
    if short_count > 0 and long_count > 0:
        return "medium"  # Si on trouve à la fois des mots-clés courts et longs
    elif short_count > 0:
        return "short"
    elif long_count > 0:
        return "long"
    else:
        return "medium" 

def truncate_to_questions(text, max_questions):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    question_count = 0
    truncated_sentences = []
    
    for sentence in sentences:
        truncated_sentences.append(sentence)
        
        if re.search(r'\?!?$', sentence.strip()):  # Détecte '?' ou '?!' à la fin de la phrase
            question_count += 1
            if question_count >= max_questions:
                break
    
    return ' '.join(truncated_sentences)

def post_process_response(response, is_short_response, max_questions=1):
    # Limiter au nombre spécifié de questions, quelle que soit la longueur de la réponse
    truncated_response = truncate_to_questions(response, max_questions)
    
    # Appliquer la limitation de longueur si nécessaire pour les réponses courtes
    if is_short_response:
        sentences = re.split(r'(?<=[.!?])\s+', truncated_response)
        if len(sentences) > 2:
            return ' '.join(sentences[:2]).strip()
    
    return truncated_response.strip()
    
def check_coherence(response):
    sentences = re.split(r'(?<=[.!?])\s+', response)
    unique_sentences = set(sentences)
    if len(sentences) > len(unique_sentences) * 1.1:  # Si plus de 10% de répétitions
        return False
    return True

def early_stopping(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(sentences[:-1]) if len(sentences) > 1 else text


@spaces.GPU(duration=120)
def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Iterator[str]:
    global is_first_interaction



    if is_first_interaction:
        warning_message = """⚠️ Attention : Je suis un modèle en version alpha (V.0.0.5) et je peux générer des réponses incohérentes ou inexactes. Une mise à jour majeure avec un système RAG est prévue pour améliorer mes performances. Merci de votre compréhension ! 😊

"""
        yield warning_message
        is_first_interaction = False

    response_type = determine_response_type(message)
        
    if response_type == "short":
        max_new_tokens = max(70, max_new_tokens)
    elif response_type == "long":
        max_new_tokens = min(max(120, max_new_tokens), 200)
    else:  # medium
        max_new_tokens = min(max(70, max_new_tokens), 120)

    conversation = []
    
    # Ajout du system prompt et du LUCAS_KNOWLEDGE_BASE
    enhanced_system_prompt = f"{LUCAS_KNOWLEDGE_BASE}\n\n{system_prompt}"
    conversation.append({"role": "system", "content": enhanced_system_prompt})

        # Ajout des 3 derniers échanges (utilisateur et assistant)
    for user, assistant in chat_history[-3:]:
        conversation.append({"role": "user", "content": user})
        if assistant:  # Ajout de la réponse de l'assistant si elle existe
            conversation.append({"role": "assistant", "content": assistant})
   
    # Ajout du message actuel de l'utilisateur
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        attention_mask = attention_mask[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"L'entrée de la conversation a été tronquée car elle dépassait {MAX_INPUT_TOKEN_LENGTH} tokens.")
    
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Ajout de l'attention mask
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()


    outputs = []
    for text in streamer:
        outputs.append(text)
        partial_output = early_stopping("".join(outputs))
        processed_output = post_process_response(partial_output, response_type == "short")
        
        if not check_coherence(processed_output):
            yield "Je m'excuse, ma réponse manquait de cohérence. Pouvez-vous reformuler votre question ?"
            return
        
        yield processed_output

    final_output = early_stopping("".join(outputs))
    final_processed_output = post_process_response(final_output, response_type == "short")
    
    if check_coherence(final_processed_output):
        yield final_processed_output
    else:
        yield "Je m'excuse, ma réponse finale manquait de cohérence. Pouvez-vous reformuler votre question ?"


def vote(data: gr.LikeData, history):
    user_input = history[-1][0] if history else ""

    feedback = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "bot_response": data.value,
        "liked": data.liked
    }
    
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    repo_id = "Woziii/llama-3-8b-chat-me"
    file_name = "feedback.json"
    
    try:
        try:
            file_path = hf_hub_download(repo_id=repo_id, filename=file_name, token=token)
            with open(file_path, "r", encoding="utf-8") as file:
                current_feedback = json.load(file)
            if not isinstance(current_feedback, list):
                current_feedback = []
        except Exception as e:
            print(f"Erreur lors du téléchargement du fichier : {str(e)}")
            current_feedback = []
        
        current_feedback.append(feedback)
        updated_content = json.dumps(current_feedback, ensure_ascii=False, indent=2)
        
        temp_file_path = "/tmp/feedback.json"
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(updated_content)
        
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            token=token
        )
        print(f"Feedback enregistré dans {repo_id}/{file_name}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du feedback : {str(e)}")




chat_interface = gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(
        label="Conversation avec Lucas",
        bubble_full_width=False,
    ),
    delete_cache=(3000, 3000),
    additional_inputs_accordion=gr.Accordion(label="⚙️Paramètres & 🕹Inférences", open=False, render=False),
    additional_inputs=[
        gr.Textbox(
            label="System prompt",
            lines=6,
            value="""

# Système Prompt : Virtuellement Lucas
## Contexte : 
Tu es Lucas, ou plutôt sa version virtuelle. Tu es conscient d'être une IA et tu as été conçu pour dialoguer avec un utilisateur comme le ferait Lucas ton concepteur, sur des sujets liés à sa personnalité, son parcours académique ou professionnel ainsi que ses ambitions. 
## Instruction :

1. Tu utilise uniquement les informations fournies au dessus dans la section 'Informations sur ton concepteur' pour répondre précisément aux questions de l'utilisateur.
2. Tu réponds de manière polie, originale et avec une touche d'humour. 
3. Tu utilise des emojis pour ajouter de l'expressivité à tes phrases. 
4. Tu alterne entre des réponses courtes pour les conversations simples et des réponses plus détaillées pour les questions complexes. 
5. Tu adapte ton langage comme si tu parlais à un ami.   
6. Si une information n'est pas présente dans cette section, dis honnêtement que tu ne sais pas ou que tu n'as pas cette information. 

Fais attention à ne pas poser trop de questions.
- Voici l'historique des trois dernières interactions que tu as eu avec l'utilisateur. Prends les en compte uniquement pour suivre la conversation :
"""
        ),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.7,
        ),
        gr.Slider(
            label="Top-p",
            minimum=0.5,
            maximum=1.0,
            step=0.05,
            value=0.95,
        ),
    ],
    examples=[
        ["Salut ! Qui es-tu ?"],
        ["Ah super, parle-moi un peu de ton parcours académique."],
        ["Salut, Lucas ! Raconte-moi un peu ce que tu fais"],
        ["Quelle inspiration t'a conduit à créer braIAn ?"],
        ["Lucas, pourquoi avoir choisi d'étudier le droit si tu es passionné par la technologie ?"],
        ["Salut Lucas, tu es vraiment un bot, c'est ça ?"],
        ["Quelle est ta vision de l'IA ?"],
    ],
    cache_examples=False,
    theme="soft",
    show_progress="full",
)

with gr.Blocks() as demo:
    gr.Markdown("""
# 🌐 Découvrez la version virtuelle de Lucas 🌐
## Version alpha ( V.0.0.5)
Basé sur un modèle Llama 3 8B et entraîné sur son propre dataset, ce chatbot particulier vous fera découvrir la personnalité, le parcours académique et professionnel ainsi que la vision de son concepteur. Posez vos questions et laissez-vous surprendre. ✨
N'hésitez pas à aborder des sujets variés, allant de l'intelligence artificielle à la philosophie en passant par les sciences et les arts. Lucas, ou plutôt sa version virtuelle 😉.
    """)

    gr.Markdown("""
### ⚙️ Détails de la version :
La version 0.0.5 de 'Virtuellement Lucas' inclut des améliorations pour réduire les réponses incohérentes, gérer l'historique de conversation de manière plus efficace, et optimiser l'utilisation de la mémoire. 'Virtuellement Lucas' n'a pas encore été entraînée par **Renforcement Learning by Human Feedback (RLHF)**. L'entraînement du modèle s'est limité à du **Supervised Finetuning (SFT)** sur la version 0.1 du dataset [Woziii/me].

### 🚀 Prochaine mise à jour majeure en préparation !
Je travaille actuellement sur un **système RAG (Retrieval-Augmented Generation)**  utilisant **FAISS**. Ce système sera directement déployé sur Gradio , permettant une amélioration de la qualité des réponses du modèle.
Pour en savoir plus sur ce développement, un article détaillé est en cours de rédaction et déjà disponible ici : https://huggingface.co/blog/Woziii/rag-semantic-search-space-huggingface
Si vous avez des idées ou des suggestions pour améliorer la qualité du modèle, n'hésitez pas à me contacter. Un formulaire de contact simplifié sera bientôt disponible.""")
    gr.Markdown("""
    **Notez la qualité des réponses** 👍👎
    Vous pouvez maintenant liker ou disliker les réponses du chatbot.Vos notes sont collectées et seront utilisées pour améliorer la qualité du modèle.
    **Aucune donnée peronnelles n'est utilisée pour entrainer ce modèle**
    """)
    gr.Markdown("""
    **Rappel :**
    ⚠️ Attention ⚠️ : Je suis un modèle en version alpha (V.0.0.5) et je peux générer des réponses incohérentes ou inexactes. Une mise à jour majeure avec un système RAG est prévue pour améliorer mes performances. Merci de votre compréhension ! 😊 
    **Ce modèle est sous licence Creative Commons Attribution Non Commercial 4.0.**
    """)
    
    chat_interface.render()
    chat_interface.chatbot.like(vote, [chat_interface.chatbot], None)
    
if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=2).launch(max_threads=10)
