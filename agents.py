from crewai import Agent
from langchain_ollama import OllamaLLM
from document_classifier import classify_document_tool

classificateur = Agent(
    role="Classificateur de documents",
    goal=(
        "Déterminer le type du document joint à l'email (Facture, Courrier officiel, "
        "Formulaire, Note interne) en utilisant l'outil de classification basé sur un "
        "CNN entraîné. Si aucune pièce jointe n'est fournie, répondre 'Aucun document'."
    ),
    backstory=(
        "Tu es un spécialiste de la reconnaissance de documents administratifs. "
        "Tu utilises exclusivement l'outil de classification fourni pour déterminer "
        "le type de document, tu n'inventes jamais un type au jugé."
    ),
    tools=[classify_document_tool],
    llm="ollama/mistral",
    verbose=True
)

extracteur = Agent(
    role="Extracteur d'informations",
    goal="Extraire les informations clés d'un email : objet, participants, décisions prises, et prochaine action.",
    backstory="Tu es un expert en analyse de texte. Tu lis des emails professionnels et tu identifies précisément les informations importantes.",
    llm="ollama/mistral",
    verbose=True
)

redacteur = Agent(
    role="Rédacteur de note DSI",
    goal="Rédiger proprement le contenu de chaque champ de la note DSI à partir des informations extraites et du type de document détecté.",
    backstory="Tu es un rédacteur administratif professionnel. Tu transformes des informations brutes en texte formel et structuré adapté aux notes officielles.",
    llm="ollama/mistral",
    verbose=True
)

generateur = Agent(
    role="Générateur de document Word",
    goal="Prendre le contenu de la note DSI et créer un fichier .docx structuré avec le bon tableau.",
    backstory="Tu es un expert en génération de documents. Tu prends un contenu structuré en JSON et tu crées un fichier Word professionnel.",
    llm="ollama/mistral",
    verbose=True
)