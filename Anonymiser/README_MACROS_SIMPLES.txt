Macros simples sans ActiveX
===========================

Ces deux modules evitent Scripting.Dictionary, donc ils ne declenchent pas
l'erreur Excel 429 "Le composant ActiveX ne peut pas creer l'objet".

Fichiers:

- AnonymiserSimple.bas
- DesanonymiserSimple.bas

Installation dans Excel:

1. Ouvrir le fichier .xlsm.
2. Ouvrir l'editeur VBA avec Alt+F11.
3. Supprimer l'ancien module qui contient Anonymiser ou Desanonymiser, si besoin.
4. File > Import File...
5. Importer le bon fichier .bas.
6. Enregistrer le .xlsm.

Utilisation:

- Dans Liste_anonymiser.xlsm, executer la macro Anonymiser.
- Dans assignments_desanonymiser.xlsm, executer la macro Desanonymiser.
- Dans assignments_desanonymiser.xlsm, la macro la plus simple est
  ImporterEtDesanonymiser:
  1. ouvrir d'abord assignments.xlsx genere par Streamlit,
  2. revenir dans assignments_desanonymiser.xlsm,
  3. lancer ImporterEtDesanonymiser,
  4. elle importe tous les onglets du fichier ouvert,
  5. elle conserve le full diccionari,
  6. elle desanonymise directement dans le fichier .xlsm ouvert.

Important:

- Anonymiser modifie directement les colonnes A, H, I, J, K du full liste.
- Si le full diccionari est vide, il le cree a partir de la colonne A.
- Ne jamais envoyer un fichier qui contient le full diccionari.
- Desanonymiser utilise le full diccionari avec:
  colonne A = nom_real
  colonne B = id_anonim
