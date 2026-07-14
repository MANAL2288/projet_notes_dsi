import os
import re
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTACHMENTS_DIR = os.path.join(BASE_DIR, "attachments")
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# Extensions qu'on accepte de faire passer dans le classificateur CNN
ALLOWED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")


def get_gmail_service():
    creds = None
    token_path = os.path.join(BASE_DIR, 'token.json')
    creds_path = os.path.join(BASE_DIR, 'credentials.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def extract_body(payload):
    body = ''
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data', '')
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    break
            elif part['mimeType'] == 'text/html' and not body:
                data = part['body'].get('data', '')
                if data:
                    html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    body = re.sub(r'<[^>]+>', ' ', html)
                    body = re.sub(r'\s+', ' ', body).strip()
    else:
        data = payload['body'].get('data', '')
        if data:
            body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    return body


def _walk_parts(parts):
    """Aplati récursivement l'arborescence de parts d'un email (les pièces
    jointes peuvent être nichées dans des sous-parts multipart/mixed)."""
    flat = []
    for part in parts:
        if 'parts' in part:
            flat.extend(_walk_parts(part['parts']))
        else:
            flat.append(part)
    return flat


def extract_attachments(service, msg_id, payload):
    """
    Parcourt le payload d'un email, télécharge les pièces jointes
    (PDF/images uniquement) et les sauvegarde sur disque.

    Retourne une liste de dicts : [{"filename": ..., "path": ..., "mimeType": ...}, ...]
    """
    attachments = []

    if 'parts' not in payload:
        return attachments

    for part in _walk_parts(payload['parts']):
        filename = part.get('filename') or ''
        body = part.get('body', {})
        attachment_id = body.get('attachmentId')
        mime_type = part.get('mimeType', '')
        print(f"[DEBUG RAW] filename={repr(filename)}, mimeType={mime_type}, attachment_id={repr(attachment_id)}")

        # On garde uniquement les vraies pièces jointes binaires (pas les parts text/html du corps)
        if not attachment_id:
            continue
        if mime_type.lower() in ("text/plain", "text/html"):
            continue

        is_allowed_ext = filename.lower().strip().endswith(ALLOWED_EXTENSIONS)
        is_allowed_mime = mime_type.lower() in (
            "application/pdf", "image/png", "image/jpeg", "image/jpg",
            "application/octet-stream"
        )
        if not filename and not is_allowed_mime:
            print(f"[SKIP] filename vide et mime non reconnu — mime={mime_type}")
            continue
        if filename and not (is_allowed_ext or is_allowed_mime):
            print(f"[SKIP] {filename} rejeté — mime={mime_type}")
            continue

        if not filename:
            ext_map = {"application/pdf": ".pdf", "image/png": ".png",
                       "image/jpeg": ".jpg", "image/jpg": ".jpg",
                       "application/octet-stream": ".pdf"}
            filename = "piece_jointe" + ext_map.get(mime_type.lower(), ".pdf")

        try:
            att = service.users().messages().attachments().get(
                userId='me', messageId=msg_id, id=attachment_id
            ).execute()
            data = att.get('data', '')
            if not data:
                continue

            file_bytes = base64.urlsafe_b64decode(data)

            if not filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                ext_map = {
                     "application/pdf": ".pdf",
                     "image/png": ".png",
                     "image/jpeg": ".jpg",
                     "image/jpg": ".jpg",
                 }
                filename = filename + ext_map.get(mime_type, ".pdf")

            safe_name = f"{msg_id}_{filename}".replace("/", "_")
            file_path = os.path.join(ATTACHMENTS_DIR, safe_name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            attachments.append({
                "filename": filename,
                "path": file_path,
                "mimeType": part.get('mimeType', '')
            })

        except Exception as e:
            # on log l'erreur mais on ne casse pas tout le pipeline
            print(f"[extract_attachments] Erreur sur {filename} : {e}")
            continue

    return attachments


def get_recent_emails(max_results=10):
    service = get_gmail_service()
    results = service.users().messages().list(
        userId='me', maxResults=max_results, labelIds=['INBOX']
    ).execute()

    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me', id=msg['id'], format='full'
        ).execute()

        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'Sans objet')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Inconnu')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), '')

        body = extract_body(msg_data['payload'])
        attachments = extract_attachments(service, msg['id'], msg_data['payload'])

        emails.append({
            'id': msg['id'],
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            'attachments': attachments,  # <-- nouveau champ
        })

    return emails