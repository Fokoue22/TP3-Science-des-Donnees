# TP3-Science-des-Donnees

\# RAPPORT DE TRAVAIL PRATIQUE 3

\## IFT799 - Science des Données

\### Analyse de Graphes Sociaux et Algorithmes de Recherche Web



---



\*\*Auteur:\*\* \[Votre Nom]  

\*\*Matricule:\*\* \[Votre Matricule]  

\*\*Cours:\*\* IFT599/IFT799 - Science des données  

\*\*Professeur:\*\* \[Nom du professeur]  

\*\*Institution:\*\* Université de Sherbrooke  

\*\*Session:\*\* Automne 2025  

\*\*Date de remise:\*\* \[Date]



---



\## RÉSUMÉ EXÉCUTIF



Ce travail pratique explore deux domaines fondamentaux de la science des données appliquée aux réseaux : l'analyse de graphes sociaux pour l'optimisation de campagnes marketing et l'implémentation d'algorithmes de ranking web.



\### Partie 1 - Analyse de Graphes Sociaux



Nous avons analysé un graphe YouTube massif contenant \*\*1,134,890 nœuds\*\* et \*\*2,987,624 arêtes\*\* représentant les relations d'amitié entre utilisateurs. L'objectif était d'identifier les influenceurs optimaux pour une campagne marketing tout en minimisant le budget.



\*\*Méthodologie:\*\*

\- Test de 2 algorithmes de détection de communautés (Louvain, LPA)

\- Test de 4 méthodes de sélection d'influenceurs (closeness, degree, betweenness, random)

\- Simulation de propagation d'influence avec probabilités réalistes (80-100%)

\- Analyse comparative de 8 approches différentes



\*\*Résultats clés:\*\*

\- Budget idéal (100% couverture): 226,978,000 CAD

\- Meilleure approche: \[À compléter après exécution]

\- Économie réalisée: \[X]% du budget idéal

\- Couverture maintenue: \[Y]%



\### Partie 2 - Application Web de Recherche



Nous avons développé une application web Flask complète implémentant:

\- \*\*PageRank:\*\* Algorithme de Google pour classer les pages par importance

\- \*\*HITS:\*\* Algorithme identifiant autorités et hubs

\- \*\*Visualisation interactive:\*\* Graphe de liens avec Plotly

\- \*\*Interface responsive:\*\* Design Bootstrap moderne



\*\*Innovations:\*\*

\- Visualisation en temps réel du graphe de liens

\- Affichage du temps d'exécution des algorithmes

\- Taille des nœuds proportionnelle aux scores

\- Support de 3 critères de ranking (PageRank, HITS-hub, HITS-autorité)



\*\*Mots-clés:\*\* Graphes sociaux, Détection de communautés, Louvain, LPA, Centralité, PageRank, HITS, Influence marketing, NetworkX, Flask, Plotly



---



\## TABLE DES MATIÈRES



1\. \[INTRODUCTION](#1-introduction)

2\. \[PARTIE 1 : ANALYSE DE GRAPHES SOCIAUX](#2-partie-1)

3\. \[PARTIE 2 : APPLICATION WEB](#3-partie-2)

4\. \[DISCUSSION](#4-discussion)

5\. \[CONCLUSION](#5-conclusion)

6\. \[RÉFÉRENCES](#6-références)

7\. \[ANNEXES](#7-annexes)



---



\## 1. INTRODUCTION



\### 1.1 Contexte et motivation



L'analyse de réseaux sociaux est devenue cruciale à l'ère du Big Data. Les plateformes comme YouTube, Facebook, et Twitter génèrent des graphes massifs où chaque utilisateur est un nœud et chaque relation est une arête. Comprendre la structure de ces réseaux permet d'optimiser les stratégies marketing, de détecter les communautés, et de modéliser la propagation d'information.



\*\*Problématiques abordées:\*\*



1\. \*\*Marketing d'influence:\*\* Comment identifier les meilleurs influenceurs pour maximiser la couverture tout en minimisant le budget?



2\. \*\*Recherche web:\*\* Comment classer des millions de pages web selon leur importance et leur pertinence?



\### 1.2 Objectifs du travail



\*\*Objectif 1 - Optimisation de campagne marketing:\*\*

\- Analyser un graphe social de plus d'1 million de nœuds

\- Détecter les communautés naturelles du réseau

\- Identifier les influenceurs optimaux par communauté

\- Simuler la propagation d'influence

\- Comparer différentes approches (8 combinaisons)

\- Recommander la meilleure stratégie selon différents critères



\*\*Objectif 2 - Implémentation d'algorithmes de ranking:\*\*

\- Développer une application web fonctionnelle

\- Implémenter PageRank (algorithme de Google)

\- Implémenter HITS (autorités et hubs)

\- Créer une visualisation interactive du graphe

\- Valider les résultats par des tests



\### 1.3 Données utilisées



\*\*Graphe YouTube:\*\*

\- \*\*Source:\*\* Stanford Network Analysis Project (SNAP)

\- \*\*Nœuds:\*\* 1,134,890 utilisateurs YouTube

\- \*\*Arêtes:\*\* 2,987,624 relations d'amitié

\- \*\*Type:\*\* Graphe non-dirigé

\- \*\*Format:\*\* Edgelist (paires de nœuds)

\- \*\*Taille:\*\* ~50 MB



\*\*Pages web (Partie 2):\*\*

\- Collectées dynamiquement par crawling

\- Seeds fournis par l'utilisateur

\- Profondeur maximale: 3 niveaux

\- Limite: 30 pages par crawl



\### 1.4 Outils et technologies



\*\*Bibliothèques Python:\*\*

\- \*\*NetworkX 3.2:\*\* Analyse de graphes, algorithmes de communautés et centralités

\- \*\*Flask 3.0:\*\* Framework web pour l'application

\- \*\*Plotly:\*\* Visualisations interactives

\- \*\*Matplotlib:\*\* Graphiques statiques

\- \*\*Pandas:\*\* Manipulation de données tabulaires

\- \*\*BeautifulSoup:\*\* Parsing HTML pour le crawling

\- \*\*NumPy:\*\* Calculs numériques



\*\*Environnement:\*\*

\- Python 3.8+

\- Jupyter Notebook pour l'analyse

\- Visual Studio Code pour le développement web



---



\## 2. PARTIE 1 : ANALYSE DE GRAPHES SOCIAUX



\### 2.1 Problématique et contexte



Une entreprise souhaite lancer une campagne marketing sur YouTube en utilisant des influenceurs. Le défi consiste à identifier les meilleurs influenceurs pour maximiser la couverture (nombre de personnes atteintes) tout en minimisant le budget.



\*\*Scénario idéal (baseline):\*\*

\- Atteindre 100% de la population (N = 1,134,890 utilisateurs)

\- Coût: 200 CAD par personne

\- Budget total: 226,978,000 CAD



\*\*Scénario réaliste (avec influenceurs):\*\*

\- Identifier des communautés dans le réseau

\- Sélectionner un influenceur par communauté

\- Simuler la propagation de l'influence

\- Coût par influenceur: 1,000 CAD (C₂)

\- Coût par personne influencée: 20 CAD (C₁)

\- Facteur de temps: η = 2



\*\*Question de recherche:\*\*  

Quelle combinaison d'algorithme de détection de communautés et de méthode de sélection d'influenceurs offre le meilleur compromis entre budget et couverture?



\### 2.2 Données et méthodologie



\#### 2.2.1 Chargement et analyse exploratoire du graphe



```python

import networkx as nx



\# Chargement du graphe

G = nx.read\_edgelist("youtube.graph.edgelist", nodetype=int)



\# Statistiques de base

N = G.number\_of\_nodes()  # 1,134,890 nœuds

m = G.number\_of\_edges()  # 2,987,624 arêtes

nb\_composantes = nx.number\_connected\_components(G)

```



\*\*Résultats de l'analyse exploratoire:\*\*



| Métrique | Valeur |

|----------|--------|

| Nombre de nœuds (N) | 1,134,890 |

| Nombre d'arêtes (m) | 2,987,624 |

| Degré moyen | 5.26 |

| Composantes connexes | \[À compléter] |

| Densité | 4.64 × 10⁻⁶ |



\*\*Interprétation:\*\*

\- Le graphe est très sparse (densité faible)

\- Chaque utilisateur a en moyenne ~5 amis

\- Le réseau présente une structure communautaire (à vérifier)



\#### 2.2.2 Calcul du budget et couverture idéaux



```python

def ideal\_budget\_and\_coverage(N, C\_0=200):

&nbsp;   return C\_0 \* N, 1.0



C\_0 = 200  # Coût par personne

ideal\_budget, ideal\_coverage = ideal\_budget\_and\_coverage(N, C\_0)

```



\*\*Résultats:\*\*

\- Budget idéal: 226,978,000 CAD

\- Couverture idéale: 100%



Ce scénario sert de baseline pour évaluer l'efficacité des approches basées sur les influenceurs.



\### 2.3 Détection de communautés



Nous avons testé deux algorithmes de détection de communautés:



\#### 2.3.1 Algorithme de Louvain



\*\*Principe:\*\*

\- Optimisation de la modularité du graphe

\- Approche hiérarchique multi-niveaux

\- Complexité: O(n log n)



\*\*Avantages:\*\*

\- Rapide sur les grands graphes

\- Produit des communautés cohésives

\- Résultats stables



\*\*Implémentation:\*\*

```python

import community as community\_louvain



def get\_communities\_louvain(G):

&nbsp;   partition = community\_louvain.best\_partition(G)

&nbsp;   communities = {}

&nbsp;   for node, comm in partition.items():

&nbsp;       communities.setdefault(comm, \[]).append(node)

&nbsp;   return communities

```



\*\*Résultats:\*\*

\- Nombre de communautés détectées: \[X]

\- Modularité: \[Y]

\- Temps d'exécution: \[Z] secondes



\#### 2.3.2 Label Propagation Algorithm (LPA)



\*\*Principe:\*\*

\- Propagation itérative d'étiquettes

\- Chaque nœud adopte l'étiquette majoritaire de ses voisins

\- Complexité: O(m)



\*\*Avantages:\*\*

\- Très rapide (linéaire)

\- Simple à implémenter

\- Pas de paramètres à ajuster



\*\*Inconvénients:\*\*

\- Résultats non déterministes

\- Peut produire des communautés déséquilibrées



\*\*Implémentation:\*\*

```python

def get\_communities\_lpa(G):

&nbsp;   lpa\_communities = nx.algorithms.community.label\_propagation\_communities(G)

&nbsp;   communities = {}

&nbsp;   for i, comm in enumerate(lpa\_communities):

&nbsp;       communities\[i] = list(comm)

&nbsp;   return communities

```



\*\*Résultats:\*\*

\- Nombre de communautés détectées: \[X]

\- Temps d'exécution: \[Y] secondes



\*\*Comparaison Louvain vs LPA:\*\*



| Critère | Louvain | LPA |

|---------|---------|-----|

| Nombre de communautés | \[X] | \[Y] |

| Temps d'exécution | \[X] sec | \[Y] sec |

| Stabilité | Élevée | Variable |

| Qualité (modularité) | Élevée | Moyenne |



\### 2.4 Sélection des influenceurs



Pour chaque communauté détectée, nous avons testé 4 méthodes de sélection d'influenceurs basées sur différentes mesures de centralité:



\#### 2.4.1 Centralité de proximité (Closeness Centrality)



\*\*Définition mathématique:\*\*

```

C\_closeness(v) = (n-1) / Σ d(v,u)

```

où d(v,u) est la distance entre v et u.



\*\*Interprétation:\*\*

\- Mesure la distance moyenne d'un nœud à tous les autres

\- Nœuds centraux peuvent atteindre rapidement l'ensemble du réseau

\- Idéal pour la propagation rapide d'information



\*\*Implémentation:\*\*

```python

closeness = nx.closeness\_centrality(community)

influencer = max(closeness, key=closeness.get)

```



\*\*Avantages:\*\*

\- Identifie les nœuds bien positionnés globalement

\- Propagation rapide de l'influence

\- Couverture potentiellement élevée



\#### 2.4.2 Centralité de degré (Degree Centrality)



\*\*Définition mathématique:\*\*

```

C\_degree(v) = deg(v) / (n-1)

```



\*\*Interprétation:\*\*

\- Mesure le nombre de connexions directes

\- Nœuds avec beaucoup d'amis

\- Portée immédiate maximale



\*\*Implémentation:\*\*

```python

degree = nx.degree\_centrality(community)

influencer = max(degree, key=degree.get)

```



\*\*Avantages:\*\*

\- Calcul très rapide (O(n))

\- Portée immédiate élevée

\- Intuitivement compréhensible



\#### 2.4.3 Centralité d'intermédiarité (Betweenness Centrality)



\*\*Définition mathématique:\*\*

```

C\_betweenness(v) = Σ (σ\_st(v) / σ\_st)

```

où σ\_st est le nombre de plus courts chemins entre s et t, et σ\_st(v) est le nombre passant par v.



\*\*Interprétation:\*\*

\- Mesure le nombre de plus courts chemins passant par un nœud

\- Nœuds "ponts" entre différentes parties du réseau

\- Contrôle du flux d'information



\*\*Implémentation:\*\*

```python

betweenness = nx.betweenness\_centrality(community)

influencer = max(betweenness, key=betweenness.get)

```



\*\*Avantages:\*\*

\- Identifie les nœuds stratégiques

\- Connecte différentes parties du réseau

\- Influence structurelle



\*\*Inconvénients:\*\*

\- Calcul coûteux (O(nm))

\- Peut être lent sur grands graphes



\#### 2.4.4 Sélection aléatoire (Random)



\*\*Principe:\*\*

\- Sélection uniforme aléatoire

\- Sert de baseline pour comparaison



\*\*Implémentation:\*\*

```python

influencer = np.random.choice(list(nodes))

```



\*\*Utilité:\*\*

\- Évaluer l'apport des méthodes basées sur la centralité

\- Quantifier l'amélioration par rapport au hasard



\### 2.5 Simulation de propagation d'influence



\#### 2.5.1 Modèle de propagation



Nous avons implémenté un modèle de propagation probabiliste:



\*\*Hypothèses:\*\*

1\. L'influenceur est le point de départ (toujours influencé)

2\. Chaque nœud a une probabilité p ∈ \[0.8, 1.0] d'être influencé

3\. L'influence se propage via les plus courts chemins

4\. Le temps de propagation est proportionnel à la distance



\*\*Algorithme:\*\*

```python

def simulate\_propagation(community, influencer, N, eta=2, C\_1=20, C\_2=1000):

&nbsp;   counter = 1  # L'influenceur

&nbsp;   temps\_propagation = 0

&nbsp;   

&nbsp;   for node in community.nodes():

&nbsp;       if node != influencer:

&nbsp;           prob = np.random.uniform(0.8, 1.0)

&nbsp;           if np.random.choice(\[0, 1], p=\[1-prob, prob]) == 1:

&nbsp;               try:

&nbsp;                   distance = nx.shortest\_path\_length(

&nbsp;                       community, source=influencer, target=node

&nbsp;                   )

&nbsp;                   counter += 1

&nbsp;                   temps\_propagation += distance

&nbsp;               except:

&nbsp;                   continue  # Nœud non atteignable

&nbsp;   

&nbsp;   budget = C\_2 + (counter - 1) \* C\_1

&nbsp;   temps\_propagation \*= eta

&nbsp;   coverage = counter / N

&nbsp;   

&nbsp;   return budget, coverage, temps\_propagation

```



\*\*Paramètres:\*\*

\- C₁ = 20 CAD (coût par personne influencée)

\- C₂ = 1,000 CAD (coût par influenceur)

\- η = 2 (facteur de temps)

\- Probabilité d'influence: 80-100%



\#### 2.5.2 Métriques calculées



Pour chaque approche, nous calculons:



1\. \*\*Budget total (CAD):\*\*

&nbsp;  ```

&nbsp;  Budget = Σ \[C₂ + (n\_influenced - 1) × C₁]

&nbsp;  ```



2\. \*\*Couverture (%):\*\*

&nbsp;  ```

&nbsp;  Couverture = (Σ n\_influenced) / N × 100

&nbsp;  ```



3\. \*\*Temps de propagation (sec):\*\*

&nbsp;  ```

&nbsp;  Temps = Σ (distance × η)

&nbsp;  ```



4\. \*\*Argent épargné (CAD):\*\*

&nbsp;  ```

&nbsp;  Épargne = Budget\_idéal - Budget\_simulé

&nbsp;  ```



5\. \*\*Couverture perdue (%):\*\*

&nbsp;  ```

&nbsp;  Perte = (Couverture\_idéale - Couverture\_simulée) × 100

&nbsp;  ```



\### 2.6 Résultats et analyse des visualisations



\#### 2.6.1 Tableau récapitulatif des résultats



Nous avons testé 8 combinaisons (2 algorithmes × 4 méthodes):



| Détection | Influence | Budget (CAD) | Couverture (%) | Temps prop. (s) | Temps exec. (s) | Nb comm. |

|-----------|-----------|--------------|----------------|-----------------|-----------------|----------|

| Louvain | Closeness | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[X] |

| Louvain | Degree | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[X] |

| Louvain | Betweenness | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[X] |

| Louvain | Random | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[X] |

| LPA | Closeness | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[Y] |

| LPA | Degree | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[Y] |

| LPA | Betweenness | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[Y] |

| LPA | Random | \[À remplir] | \[À remplir] | \[À remplir] | \[À remplir] | \[Y] |



\*Note: Les valeurs seront remplies après l'exécution complète du notebook\*





