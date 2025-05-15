# 🎓 Constitution Automatique des Classes

Bienvenue dans le projet **Classes Assignment**, une application Streamlit permettant de répartir automatiquement des élèves dans différentes classes selon plusieurs critères pédagogiques et sociaux.

🌐 **Application en ligne** : [https://classesassignment.streamlit.app](https://classesassignment.streamlit.app)

---

## 📝 Description du projet

Cette application facilite l’affectation automatique des élèves dans des classes, en tenant compte de leurs préférences, options choisies, niveaux scolaires, comportements, et souhaits sociaux (élèves avec/sans lesquels ils préfèrent être).

---

## 📄 Format du fichier Excel d'entrée

Le fichier Excel fourni doit contenir **2 feuilles** distinctes :

### 🧑‍🎓 Feuille `liste` : Élèves à affecter

| Colonne           | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `Elèves à affecter` | Nom unique de chaque élève                                       |
| `Genre`           | `F` (fille) ou `G` (garçon)                                        |
| `por`, `lat`, `pp` | 1 si l’élève souhaite suivre cette option, sinon 0               |
| `Niveau`          | Niveau scolaire (ex : 1, 2, 3)                                     |
| `Comportement`    | 1 (bon) à 3 (difficile)                                            |
| `avec1`, `avec2`  | (facultatif) Élèves souhaités                                     |
| `sans1`, `sans2`  | (facultatif) Élèves à éviter                                      |

### 🏫 Feuille `classes` : Classes disponibles

| Colonne    | Description                                                            |
|------------|------------------------------------------------------------------------|
| `Nom`      | Nom de la classe (ex : A, B, C)                                        |
| `por`, `lat`, `pp` | 1 si la classe propose cette option (sinon vide)            |
| `capacité` | (facultatif) Capacité maximale d’élèves dans cette classe             |

💡 Un fichier exemple peut être téléchargé depuis l'application.

---

## 📤 Structure du fichier Excel de sortie

Après traitement, le système génère un fichier avec plusieurs feuilles :

### 🏫 Feuille `Classes` : Affectations finales
| Colonne        | Description                            |
|----------------|----------------------------------------|
| `student`      | Nom de l’élève                         |
| `Genre`        | F ou G                                 |
| `por`, `lat`, `pp` | Options choisies (0 ou 1)          |
| `level`        | Niveau scolaire                        |
| `Comportement` | Indice de priorité (1 = prioritaire)  |
| `avec1`, `avec2`, `sans1`, `sans2` | Souhaits sociaux    |
| `classe`       | Classe attribuée à l’élève             |

### ❗ Feuille `Impossibilites`
Liste des contraintes non respectées.

| Colonne  | Description                      |
|----------|----------------------------------|
| `Type`   | Type de souhait (`avec1`, etc.)  |
| `Source` | Élève à l’origine du souhait     |
| `Other`  | Élève concerné                   |

### ⚠️ Feuille `Contraintes`
Liste des contraintes effectivement prises en compte.

| Colonne  | Description                      |
|----------|----------------------------------|
| `Type`   | Type de contrainte               |
| `Source` | Élève à l’origine                |
| `Other`  | Élève ciblé                      |

### 📊 Feuille `Tableau`
Vue matricielle du résultat (genre, niveau, classe).

### 📈 Feuille `Dashboards`
Statistiques agrégées par classe.

| Colonne          | Description                        |
|------------------|------------------------------------|
| `classe`         | Nom de la classe                   |
| `Total`          | Nombre total d’élèves              |
| `Niveau1-3`      | Répartition par niveau             |
| `POR`, `LAT`     | Répartition par option             |
| `Filles`, `Garçons` | Répartition par genre          |
| `Comp1-3`        | Niveau de comportement             |
| `%N1-N3`, etc.   | Pourcentages divers                |

---

## 🛠️ Installation locale

```bash
git clone https://github.com/jordiordonez/classes_assignment.git
cd classes_assignment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # si disponible
streamlit run app.py  # ou le fichier principal
```

---

## 👤 Auteur

- Jordi Ordoñez Adellach  
- [GitHub - jordiordonez](https://github.com/jordiordonez)

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus d'informations.

