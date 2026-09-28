
# MCDM Studio — Application Streamlit

Application pédagogique d'aide à la décision multicritère basée sur le cours fourni.

## Scénario

Choisir le meilleur site pour implanter un centre logistique.

Le scénario par défaut contient :
- 10 alternatives ;
- 8 critères ;
- des critères bénéfice (+) et coût (-) ;
- des données fictives destinées à la démonstration.

L'application permet aussi :
- d'ajouter/supprimer des alternatives ;
- d'ajouter/supprimer des critères ;
- de modifier le sens d'un critère ;
- d'importer un CSV.

## Méthodes implémentées

### Pondération
- AHP
- BWM
- Entropie
- CRITIC
- Poids égaux (outil pratique, non présenté comme méthode du cours)

### Classement
- WSM
- WPM
- WASPAS
- TOPSIS
- VIKOR

### Robustesse / comparaison
- Corrélation de Spearman
- Analyse de sensibilité

## Méthodes citées mais non implémentées

DEMATEL, SWARA, ELECTRE, PROMETHEE, ANP et Fuzzy MCDM sont mentionnées dans le PDF
mais ne sont pas suffisamment détaillées dans cette partie pour être implémentées fidèlement.

## Choix d'implémentation explicite

Pour l'Entropie, le PDF donne la formule à partir d'une matrice normalisée mais
ne donne pas, sur ces diapositives, une formule de normalisation propre à l'Entropie.
L'application utilise donc une normalisation min-max bénéfice/coût avant le calcul de l'Entropie.

## Installation

```bash
python -m venv .venv
```

Windows :
```bash
.venv\Scripts\activate
```

macOS/Linux :
```bash
source .venv/bin/activate
```

Puis :
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Format d'un CSV importé

La première colonne contient les noms des alternatives.

Exemple :

```text
Alternative,Cout,Qualite,Delai
A1,100,8,12
A2,80,7,15
A3,90,9,10
```

Après import, régler le sens (+/-) de chaque critère dans l'application.
