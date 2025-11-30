import os
import uuid
from flask import Flask, request, jsonify, render_template
from concurrent.futures import ThreadPoolExecutor
from main4 import process_resistor_image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Pool de threads pour les tâches longues
executor = ThreadPoolExecutor(max_workers=2)
jobs = {}

@app.route('/')
def index():
    return render_template("resistance_analyzer.html")


def long_running_analysis(job_id, filepath):
    """
    Cette fonction s'exécute en arrière-plan.
    Elle est totalement indépendante de la requête HTTP initiale.
    """
    try:
        print(f"[{job_id}] Début du traitement sur {filepath}")

        # --- SIMULATION DU TRAITEMENT LOURD (ex: IA, OCR) ---
        result = process_resistor_image(
            img_path=filepath,
            out_dir="debug_out",
        )


        # Exemple d'utilisation réelle :
        # result = mon_programme_ia.predict(filepath)

        # Résultat factice pour l'exemple
        #result = {
        #    "resistance": "4.7k",
        #    "unit": "Ω",
        #    "tolerance": "±1%",
        #    "colors": ["Jaune", "Violet", "Rouge", "Marron"]
        #}

        jobs[job_id] = {"state": "done", "result": result}
        # print(f"[{job_id}] Traitement terminé avec succès")

    except Exception as e:
        print(f"[{job_id}] Erreur : {e}")
        jobs[job_id] = {"state": "failed", "error": str(e)}

    finally:
        # --- NETTOYAGE DU FICHIER ---
        # Le 'finally' garantit que le fichier est supprimé
        # même si le code plante dans le bloc 'try'
        if os.path.exists(filepath):
            os.remove(filepath)
            # print(f"[{job_id}] Fichier temporaire supprimé")


@app.route('/analyze-async', methods=['POST'])
def analyze_async():
    if 'photo' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    # 1. Générer l'ID unique pour cette tâche
    job_id = str(uuid.uuid4())

    # 2. Construire un nom de fichier sécurisé basé sur l'UUID
    # On garde l'extension d'origine (.jpg, .png) pour que les outils d'image s'y retrouvent
    _, ext = os.path.splitext(file.filename)
    filename = f"{job_id}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # 3. Sauvegarder physiquement le fichier MAINTENANT
    # C'est indispensable car 'file' sera détruit à la fin de cette fonction
    file.save(filepath)

    # 4. Initialiser l'état du job
    jobs[job_id] = {"state": "processing"}

    # 5. Lancer le thread avec le CHEMIN vers le fichier sauvegardé
    executor.submit(long_running_analysis, job_id, filepath)

    # 6. Répondre immédiatement au client avec le ticket (202 Accepted)
    return jsonify({
        "message": "Image reçue, traitement démarré",
        "job_id": job_id
    }), 202


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify(job)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
