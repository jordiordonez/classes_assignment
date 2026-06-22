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
| `Comportement` | Niveau de comportement (1 = bon à 3 = difficile) |
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

## ✏️ Modification manuelle des affectations

Une fois l'affectation automatique réalisée, vous pouvez :

1. Modifier les classes dans le tableau affiché dans l'application
2. Cliquer sur **"🔁 Rafraîchir et vérifier les contraintes"** pour mettre à jour toutes les feuilles
3. Télécharger un **nouveau fichier Excel** prenant en compte vos modifications

Cela permet d'adapter finement l'affectation en tenant compte d'éléments non formalisés dans les données initiales.


---

## 🔒 Données personnelles et anonymisation

L'application peut traiter des données personnelles d'élèves. Pour un usage avec
l'application en ligne, il est recommandé de ne jamais envoyer les noms réels.

Deux options sont possibles :

1. Exécuter l'application localement sur un ordinateur autorisé de l'établissement.
2. Utiliser les fichiers Excel du dossier `Anonymiser` pour remplacer les noms par
   des identifiants anonymes avant l'envoi, puis rétablir les noms après traitement.

Le fichier `diccionari` contient la correspondance entre noms réels et identifiants
anonymes. Il doit rester local et ne doit jamais être envoyé à l'application en ligne.

---

## 🕶️ Procédure d'anonymisation Excel

Le dossier `Anonymiser` contient deux fichiers :

- `Liste_anonymiser.xlsm`
- `assignments_desanonymiser.xlsm`

### 1. Anonymiser le fichier d'entrée

1. Ouvrir `Liste_anonymiser.xlsm`.
2. Copier ou compléter les données dans la feuille `liste`.
3. Vérifier que la feuille `diccionari` existe.
4. Cliquer sur le bouton ou lancer la macro `Anonymiser`.
5. La macro remplace directement les noms par les identifiants anonymes dans :
   - `Elèves à affecter`
   - `avec1`
   - `avec2`
   - `sans1`
   - `sans2`
6. Créer ensuite une copie `.xlsx` destinée à l'application en ligne, sans la feuille
   `diccionari`.

Le fichier envoyé à Streamlit doit contenir seulement les données anonymisées et les
feuilles nécessaires (`liste` et `classes`). Il ne doit pas contenir le dictionnaire.

### 2. Désanonymiser le fichier de sortie

Après traitement, Streamlit génère un fichier `assignments.xlsx` contenant les
identifiants anonymes.

Procédure pratique :

1. Ouvrir `assignments.xlsx`.
2. Ouvrir `assignments_desanonymiser.xlsm`.
3. Dans `assignments.xlsx`, sélectionner tous les onglets :
   - cliquer sur le premier onglet,
   - maintenir `Shift`,
   - cliquer sur le dernier onglet.
4. Clic droit sur un onglet sélectionné.
5. Choisir `Déplacer ou copier...`.
6. Dans `Dans le classeur`, choisir `assignments_desanonymiser.xlsm`.
7. Cocher `Créer une copie`.
8. Valider.
9. Dans `assignments_desanonymiser.xlsm`, vérifier que la feuille `diccionari` est
   présente.
10. Lancer la macro `Desanonymiser`.

La macro remplace les identifiants anonymes par les noms réels dans les feuilles de
résultat copiées.

---

## 🛠️ Installation locale

```bash
git clone https://github.com/jordiordonez/classes_assignment.git
cd classes_assignment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Auteur

- Jordi Ordoñez Adellach  
- [GitHub - jordiordonez](https://github.com/jordiordonez)

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](https://github.com/jordiordonez/classes_assignment/blob/main/LICENCE) pour plus d'informations.
