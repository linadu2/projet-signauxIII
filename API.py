from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

# Création de l'application
app = Flask(__name__)

# Dossier où on va sauvegarder temporairement les images reçues
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- TA FONCTION DE TRAITEMENT ---
def mon_programme_resistance(chemin_image):
    """
    C'est ici que tu intègres ton script existant.
    Pour l'instant, c'est une simulation.
    """
    print(f"Analyse de l'image : {chemin_image}")
    
    # TODO: Remplace ceci par ton vrai appel de fonction
    # ex: valeur, couleurs = detecter_resistor(chemin_image)
    
    # Simulation du résultat
    resultat = {
        "resistance": "4.7k",
        "unit": "Ω",
        "tolerance": "±1%",
        "colors": ["Jaune", "Violet", "Rouge", "Marron"]
    }
    return resultat

@app.route('/')
def index():
    return render_template("resistance_analyzer.html")

# --- LA ROUTE UNIQUE (Réception + Réponse) ---
@app.route('/analyze', methods=['POST'])
def analyze_image():
    # 1. Vérifier si une image est présente dans la requête
    if 'photo' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
    
    file = request.files['photo']
    
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    if file:
        # 2. Sauvegarder l'image en toute sécurité
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 3. Appeler ton programme Python
            data = mon_programme_resistance(filepath)
            
            # 4. Nettoyage (optionnel) : supprimer l'image après analyse
            os.remove(filepath)
            
            # 5. Renvoyer le résultat en JSON au téléphone
            return jsonify(data)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # host='0.0.0.0' est CRUCIAL pour que ton téléphone puisse accéder au PC
    app.run(host='0.0.0.0', port=5000)
