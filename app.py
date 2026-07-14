import streamlit as st
from crewai import Crew, Process
from agents import extracteur, redacteur, generateur, classificateur
from tasks import create_tasks
from utils import generate_docx, parse_json_from_text
from gmail_service import get_recent_emails
from document_classifier import classify_document_file

st.set_page_config(
    page_title="Générateur de Notes DSI",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Générateur de Notes DSI")
st.markdown("Connectez votre Gmail, sélectionnez un email et générez automatiquement une note DSI.")

st.divider()

# ─── ÉTAPE 1 : Chargement des emails Gmail ───────────────────────────────────
if "emails" not in st.session_state:
    st.session_state["emails"] = []
if "email_selectionne" not in st.session_state:
    st.session_state["email_selectionne"] = None

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📬 Boîte Gmail")
with col2:
    if st.button("🔄 Charger les emails", use_container_width=True):
        with st.spinner("Connexion à Gmail..."):
            try:
                st.session_state["emails"] = get_recent_emails(max_results=10)
                for e in st.session_state["emails"]:
                    print(f"[DEBUG] {e['subject']} -> attachments: {e.get('attachments')}")
                st.success(f"{len(st.session_state['emails'])} emails chargés !")
            except Exception as e:
                st.error(f"Erreur Gmail : {e}")

# ─── ÉTAPE 2 : Liste des emails ──────────────────────────────────────────────
if st.session_state["emails"]:
    st.markdown("**Sélectionnez un email à transformer en note :**")

    for i, email in enumerate(st.session_state["emails"]):
        with st.container(border=True):
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{email['subject']}**")
                st.caption(f"De : {email['sender']}  |  {email['date'][:25]}")
                if email.get("attachments"):
                    noms = ", ".join(a["filename"] for a in email["attachments"])
                    st.caption(f"📎 Pièces jointes : {noms}")
            with col_btn:
                if st.button("Sélectionner", key=f"sel_{i}", use_container_width=True):
                    st.session_state["email_selectionne"] = email
                    st.session_state["generated"] = False
                    st.session_state["classification_result"] = None

# ─── ÉTAPE 3 : Aperçu de l'email sélectionné ────────────────────────────────
if st.session_state["email_selectionne"]:
    email = st.session_state["email_selectionne"]
    st.divider()
    st.subheader("✉️ Email sélectionné")

    with st.expander("Voir le contenu de l'email", expanded=False):
        st.markdown(f"**Objet :** {email['subject']}")
        st.markdown(f"**De :** {email['sender']}")
        st.markdown(f"**Date :** {email['date']}")
        st.text(email['body'][:1000])

    # Chemin de la première pièce jointe (PDF ou image), s'il y en a une
    attachment_path = None
    if email.get("attachments"):
        attachment_path = email["attachments"][0]["path"]
        st.info(f"📎 Pièce jointe détectée : {email['attachments'][0]['filename']}")
    else:
        st.caption("Aucune pièce jointe sur cet email.")

    if st.button("🚀 Générer la note DSI", use_container_width=True, type="primary"):
        with st.spinner("Les agents analysent l'email..."):
            try:
                # ─── Classification réelle, appelée DIRECTEMENT en Python ───
                # On n'utilise plus la sortie du LLM (tache_classification.output.raw)
                # pour l'affichage, car Mistral/Ollama a tendance à reformuler ou
                # halluciner par-dessus le résultat réel de l'outil CNN.
                if attachment_path:
                    try:
                        classif_reelle = classify_document_file(attachment_path)
                        st.session_state["classification_result"] = (
                            f"{classif_reelle['label']} "
                            f"(confiance : {classif_reelle['confidence'] * 100:.1f}%)"
                        )
                    except Exception as e:
                        st.session_state["classification_result"] = f"Erreur classification : {e}"
                else:
                    st.session_state["classification_result"] = "Aucun document joint"

                email_text = f"""
De : {email['sender']}
Objet : {email['subject']}
Date : {email['date']}

{email['body']}
"""
                tache_classification, tache_extraction, tache_redaction, tache_generation = create_tasks(
                    email_text, attachment_path=attachment_path
                )

                crew = Crew(
                    agents=[classificateur, extracteur, redacteur, generateur],
                    tasks=[tache_classification, tache_extraction, tache_redaction, tache_generation],
                    process=Process.sequential,
                    verbose=True
                )

                result = crew.kickoff()

                redaction_output = tache_redaction.output.raw if tache_redaction.output else ""
                if not redaction_output or "Non déterminé" in str(parse_json_from_text(redaction_output).get("cadre", "")):
                    redaction_output = tache_extraction.output.raw if tache_extraction.output else str(result)

                data = parse_json_from_text(redaction_output)
                st.session_state["data"] = data
                st.session_state["docx_bytes"] = generate_docx(data)
                st.session_state["generated"] = True

            except Exception as e:
                st.error(f"Erreur : {e}")

# ─── ÉTAPE 4 : Aperçu et téléchargement ─────────────────────────────────────
if st.session_state.get("generated"):
    st.divider()
    st.subheader("📋 Aperçu de la note générée")

    # Affichage du résultat RÉEL du CNN (appelé directement, pas via le LLM) —
    # c'est ici le checkpoint "humain-dans-la-boucle" : l'utilisateur voit ce
    # que le modèle a détecté avant de valider/télécharger.
    if st.session_state.get("classification_result"):
        st.markdown("**🧠 Type de document détecté (par le modèle CNN) :**")
        st.warning(st.session_state["classification_result"])

    data = st.session_state["data"]

    fields = [
        ("🏢 Cadre", "cadre"),
        ("📌 Objet", "objet"),
        ("👥 Participants", "participants"),
        ("📝 Descriptif", "descriptif"),
        ("⚡ Prochaine Action", "prochaine_action"),
    ]

    for label, key in fields:
        st.markdown(f"**{label}**")
        st.info(data.get(key, "—"))

    st.divider()
    st.download_button(
        label="⬇️ Télécharger la note (.docx)",
        data=st.session_state["docx_bytes"],
        file_name="Note_DSI.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
        type="primary"
    )