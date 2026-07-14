from crewai import Task
from agents import classificateur, extracteur, redacteur, generateur


def create_tasks(email_text, attachment_path=None):

    tache_classification = Task(
        description=f"""
Tu dois déterminer le type du document joint à cet email, s'il y en a un.

Chemin du fichier joint : {attachment_path if attachment_path else "AUCUN"}

RÈGLES :
1. Si un chemin de fichier est fourni, utilise l'outil de classification de documents
   avec ce chemin exact.
2. Si aucun fichier n'est fourni (AUCUN), réponds simplement "Aucun document joint".
3. Ne réponds qu'avec le résultat de l'outil ou "Aucun document joint", rien d'autre.
""",
        expected_output="Le type de document détecté (et sa confiance) ou 'Aucun document joint'.",
        agent=classificateur
    )

    tache_extraction = Task(
        description=f"""
Tu es un extracteur d'informations. Analyse cet email et réponds UNIQUEMENT avec du JSON.

EMAIL À ANALYSER :
{email_text}

Tu disposes aussi du type de document détecté par le classificateur (voir contexte).
Utilise cette information pour renseigner le champ "cadre" si c'est pertinent
(par exemple si le document est une "Facture", le cadre peut être "Financier").

RÈGLES ABSOLUES :
1. Réponds SEULEMENT avec le JSON, rien d'autre
2. Le "descriptif" = résumé DÉTAILLÉ du contenu (3-4 phrases) :
   mentionne les informations clés comme le RIB, les documents joints,
   les instructions données dans l'email
3. La "prochaine_action" = action concrète à faire suite à cet email
   (ne jamais laisser vide)
4. Les "participants" = noms/emails présents dans l'email uniquement
5. NE JAMAIS copier les instructions dans le JSON

JSON à retourner :
{{
    "cadre": "...",
    "objet": "...",
    "participants": "...",
    "descriptif": "...",
    "prochaine_action": "..."
}}
""",
        expected_output="JSON valide uniquement, 5 champs string.",
        agent=extracteur,
        context=[tache_classification]
    )

    tache_redaction = Task(
        description="""
Tu reçois un JSON avec des informations extraites d'un email.
Reformule chaque valeur en français formel et professionnel.
Tu dois répondre UNIQUEMENT avec du JSON valide, sans aucun texte autour.
Toutes les valeurs doivent être des strings simples, jamais des listes ou objets.
Ne jamais laisser un champ vide.

Format JSON STRICT :
{
    "cadre": "...",
    "objet": "...",
    "participants": "...",
    "descriptif": "...",
    "prochaine_action": "..."
}
""",
        expected_output="Un objet JSON valide avec exactement 5 champs string en français.",
        agent=redacteur,
        context=[tache_extraction]
    )

    tache_generation = Task(
        description="""
Tu reçois un JSON avec le contenu final de la note DSI.
Retourne simplement le JSON reçu tel quel, sans modification.
""",
        expected_output="Le JSON reçu retourné tel quel.",
        agent=generateur,
        context=[tache_redaction]
    )

    return tache_classification, tache_extraction, tache_redaction, tache_generation