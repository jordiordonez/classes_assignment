Execution locale de l'application Classes
========================================

Contenu du dossier:

- app.py
- assign_classes_module.py
- report_module.py
- update_classes_module.py
- requirements.txt
- Liste.xlsx
- run_local_windows.bat

Utilisation sur Windows:

1. Ouvrir ce dossier sur le PC local.
2. Double-cliquer sur run_local_windows.bat.
3. Au premier lancement, le script cree un venv et installe les dependances.
4. Quand Streamlit demarre, ouvrir:

   http://localhost:8501

5. Pour un acces depuis un autre PC du meme reseau, utiliser l'URL Network URL
   affichee par Streamlit, ou l'adresse IP du PC suivie de :8501.

Remarques:

- Le premier lancement peut necessiter Internet pour installer les dependances.
- Si Python n'est pas installe, le script tente une installation avec winget.
- Si winget ou l'installation sont bloques, demander a l'administrateur
  d'installer Python 3.12 ou 3.13 avec l'option Add Python to PATH.
- Les fichiers Excel traites restent sur le PC local ou sur le reseau local.
