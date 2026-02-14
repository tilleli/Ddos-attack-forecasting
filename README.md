# 🛡️ Prédiction d'Attaques DDoS par Apprentissage Temporel

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Anticiper les attaques DDoS 1 à 10 secondes avant qu'elles ne surviennent**

</div>

---

##  Problématique

Les attaques par déni de service distribué (DDoS) saturent les serveurs cibles en générant un volume massif de requêtes malveillantes provenant d'un réseau de machines compromises (botnet). Ces attaques rendent les services indisponibles pour les utilisateurs légitimes.

**Le problème des solutions actuelles** :

Les systèmes de sécurité traditionnels (IDS, pare-feu, antivirus) fonctionnent en mode **réactif**. Ils détectent l'attaque uniquement lorsqu'elle est déjà en cours, c'est-à-dire quand les serveurs sont déjà saturés. À ce moment-là, le service subit déjà une dégradation ou une interruption complète, et il reste très peu de temps pour réagir efficacement.

**Le besoin** :

Disposer d'un système capable de **prédire** l'imminence d'une attaque DDoS quelques secondes à l'avance. Ces quelques secondes d'anticipation permettraient d'activer préventivement les mécanismes de défense, d'allouer des ressources supplémentaires et de minimiser l'impact sur les utilisateurs.

---

##  Approche

### Notre solution

Nous proposons un **système prédictif** basé sur l'intelligence artificielle qui analyse le trafic réseau en continu et détecte les **patterns précurseurs** d'une attaque DDoS.

**L'idée centrale** : Les attaques DDoS ne démarrent pas instantanément. Elles présentent des signatures caractéristiques dans les secondes qui précèdent leur pic : augmentation progressive du volume de trafic, modification des patterns de connexion, homogénéisation des requêtes, etc.

**Notre approche** :
- Analyser le trafic réseau comme une **série temporelle** (et non comme des événements isolés)
- Utiliser les données des 30 dernières secondes pour prédire l'état du système dans 1, 5 ou 10 secondes
- Transformer le problème de détection en un problème de **prédiction temporelle**

 
### Comment créer des séquences temporelles ?

C'est le cœur de notre approche. Au lieu d'analyser chaque seconde de manière isolée, nous créons des **fenêtres d'observation** qui capturent l'évolution du trafic.

#### Principe de la fenêtre glissante

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#667eea','secondaryColor':'#764ba2'}}}%%
graph LR
    subgraph Window1["Fenêtre 1"]
        A0[t0] --> A1[t1] --> A2[t2] --> A3[t3] --> A4[t4] --> A5[t5] --> A6[t6] --> A7[t7] --> A8[t8] --> A9[t9]
    end
    
    Window1 -.->|Prédiction| P1[Label t14]
    
    style Window1 fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style P1 fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000
```

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#667eea','secondaryColor':'#764ba2'}}}%%
graph LR
    subgraph Window2["Fenêtre 2 (décalée de +1)"]
        B1[t1] --> B2[t2] --> B3[t3] --> B4[t4] --> B5[t5] --> B6[t6] --> B7[t7] --> B8[t8] --> B9[t9] --> B10[t10]
    end
    
    Window2 -.->|Prédiction| P2[Label t15]
    
    style Window2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style P2 fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
```

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#667eea','secondaryColor':'#764ba2'}}}%%
graph LR
    subgraph Window3["Fenêtre 3 (décalée de +1)"]
        C2[t2] --> C3[t3] --> C4[t4] --> C5[t5] --> C6[t6] --> C7[t7] --> C8[t8] --> C9[t9] --> C10[t10] --> C11[t11]
    end
    
    Window3 -.->|Prédiction| P3[Label t16]
    
    style Window3 fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
    style P3 fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
```

#### Calcul détaillé avec exemple

Prenons un exemple concret avec :
- **seq_length** = 10 (fenêtre de 10 secondes)
- **horizon** = 5 (prédiction 5 secondes à l'avance)

**Pour la première séquence (position i=0)** :

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#06b6d4','secondaryColor':'#3b82f6'}}}%%
flowchart LR
    subgraph Input["📥 ENTRÉE (X)"]
        direction TB
        I0["t=0<br/>Données réseau"]
        I1["t=1"]
        I2["t=2"]
        I3["t=3"]
        I4["t=4"]
        I5["t=5"]
        I6["t=6"]
        I7["t=7"]
        I8["t=8"]
        I9["t=9"]
    end
    
    Input ==>|10 secondes<br/>d'observation| Model["🤖 MODÈLE IA"]
    
    Model ==>|Prédiction pour| Output["📤 SORTIE (y)<br/><br/>Label au temps<br/>t=14<br/><br/>(9 + 5 = 14)"]
    
    style Input fill:#e1f5ff,stroke:#0288d1,stroke-width:3px
    style Model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style Output fill:#fff3e0,stroke:#f57c00,stroke-width:3px
```

**Calcul** :
- Entrée : `data[0:10]` → données des temps 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Sortie : `label[0 + 10 + 5 - 1]` = `label[14]`
- **Interprétation** : En observant le trafic de t=0 à t=9, on prédit l'état à t=14

**Pour la deuxième séquence (position i=1)** :

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#06b6d4','secondaryColor':'#3b82f6'}}}%%
flowchart LR
    subgraph Input2["📥 ENTRÉE (X)"]
        direction TB
        J1["t=1"]
        J2["t=2"]
        J3["t=3"]
        J4["t=4"]
        J5["t=5"]
        J6["t=6"]
        J7["t=7"]
        J8["t=8"]
        J9["t=9"]
        J10["t=10"]
    end
    
    Input2 ==>|10 secondes<br/>d'observation| Model2["🤖 MODÈLE IA"]
    
    Model2 ==>|Prédiction pour| Output2["📤 SORTIE (y)<br/><br/>Label au temps<br/>t=15<br/><br/>(10 + 5 = 15)"]
    
    style Input2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style Model2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style Output2 fill:#ffccbc,stroke:#d84315,stroke-width:3px
```

**Calcul** :
- Entrée : `data[1:11]` → données des temps 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Sortie : `label[1 + 10 + 5 - 1]` = `label[15]`

#### Formule générale

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#8b5cf6','secondaryColor':'#ec4899'}}}%%
graph TD
    A["Pour chaque position i"] --> B["X[i] = data[i : i+seq_length]<br/><br/>📊 Fenêtre d'observation"]
    B --> C["y[i] = label[i + seq_length + horizon - 1]<br/><br/>🎯 Label futur à prédire"]
    
    C --> D["Résultat final :<br/><br/>X shape = (n_samples, seq_length, n_features)<br/>y shape = (n_samples,)"]
    
    style A fill:#fce7f3,stroke:#be185d,stroke-width:2px
    style B fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style C fill:#fed7aa,stroke:#c2410c,stroke-width:2px
    style D fill:#bbf7d0,stroke:#15803d,stroke-width:2px
```

**Ce que le modèle apprend** :

Le modèle analyse ces séquences et apprend à détecter :
- Les **variations temporelles** : augmentation progressive du trafic
- Les **patterns répétitifs** : rafales de requêtes similaires
- Les **anomalies séquentielles** : changements soudains dans les métriques réseau

---

## 🏗️ Architecture

### Pipeline complet du projet
![Schéma d'architecture](Diagramme.svg)
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#667eea','secondaryColor':'#764ba2'}}}%%
flowchart LR
    subgraph DATA["📦 DONNÉES<br/>─────────"]
        D1["CICIDS2017<br/>(Trafic réaliste)"]
        D2["CICDDOS2019<br/>(12 types DDoS)"]
    end
    
    subgraph CLEAN["🧹 PREPROCESSING<br/>─────────"]
        C1["• Nettoyage<br/>• Label binaire<br/>• Tri chrono"]
    end
    
    subgraph AGG["⏱️ AGGREGATION<br/>─────────"]
        A1["• Par seconde<br/>• Feature selection"]
    end
    
    subgraph SPLIT["✂️ SPLIT TEMPOREL<br/>─────────"]
        S1["Train: 70%<br/>(Passé)"]
        S2["Test: 30%<br/>(Futur)"]
    end
    
    subgraph NORM["📊 NORMALISATION<br/>─────────"]
        N1["RobustScaler<br/>(Anti-outliers)"]
    end
    
    subgraph SEQ["🔄 SÉQUENCES<br/>─────────"]
        Q1["Fenêtre glissante<br/>seq_length=10"]
        Q2["H=1 sec"]
        Q3["H=5 sec"]
        Q4["H=10 sec"]
    end
    
    subgraph MODEL["🤖 MODÈLE<br/>─────────"]
        M1["XGBoost<br/>300 arbres<br/>depth=8<br/>lr=0.05"]
    end
    
    subgraph PRED["✅ PRÉDICTION<br/>─────────"]
        P1["Accuracy: 97%<br/>Precision: 96%<br/>Recall: 94%"]
    end
    
    DATA ==> CLEAN
    CLEAN ==> AGG
    AGG ==> SPLIT
    SPLIT ==> NORM
    NORM ==> SEQ
    SEQ ==> MODEL
    MODEL ==> PRED
    
    style DATA fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    style CLEAN fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style AGG fill:#f3e5f5,stroke:#6a1b9a,stroke-width:3px
    style SPLIT fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style NORM fill:#fff9c4,stroke:#f9a825,stroke-width:3px
    style SEQ fill:#fce4ec,stroke:#c2185b,stroke-width:3px
    style MODEL fill:#e0f2f1,stroke:#00695c,stroke-width:3px
    style PRED fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px
```

### Détail de chaque étape

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#06b6d4','secondaryColor':'#8b5cf6'}}}%%
graph LR
    A[📦 CICIDS2017<br/>CICDDOS2019] -->|CSV files| B[🧹 Cleaning<br/>Remove NaN/Inf<br/>Binary labels]
    B -->|Clean data| C[⏱️ Time Aggregation<br/>Group by second<br/>Calculate stats]
    C -->|Aggregated| D[✂️ Temporal Split<br/>Train 70%<br/>Test 30%]
    D -->|Train/Test| E[📊 RobustScaler<br/>Fit on train<br/>Transform all]
    E -->|Normalized| F[🔄 Sequences<br/>Sliding windows<br/>Multi-horizon]
    F -->|Sequences| G[🤖 XGBoost Training<br/>n_estimators=300<br/>max_depth=8]
    G -->|Trained model| H[✅ Predictions<br/>& Evaluation]
    
    style A fill:#dbeafe,stroke:#0284c7,stroke-width:2px
    style B fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style C fill:#e9d5ff,stroke:#9333ea,stroke-width:2px
    style D fill:#d1fae5,stroke:#059669,stroke-width:2px
    style E fill:#fed7aa,stroke:#ea580c,stroke-width:2px
    style F fill:#fbcfe8,stroke:#be185d,stroke-width:2px
    style G fill:#ccfbf1,stroke:#0d9488,stroke-width:2px
    style H fill:#bbf7d0,stroke:#16a34a,stroke-width:2px
```

### Focus sur la création des 3 datasets

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#f59e0b','secondaryColor':'#ef4444'}}}%%
flowchart TB
    A["📊 Données normalisées<br/>avec timestamps"] --> B["🔄 Fonction create_sequences_multi_horizon"]
    
    B --> C1["Dataset H=1<br/>─────────<br/>X_train: (n, 10, features)<br/>y_train: label t+1<br/>X_test: (m, 10, features)<br/>y_test: label t+1"]
    
    B --> C2["Dataset H=5<br/>─────────<br/>X_train: (n, 10, features)<br/>y_train: label t+5<br/>X_test: (m, 10, features)<br/>y_test: label t+5"]
    
    B --> C3["Dataset H=10<br/>─────────<br/>X_train: (n, 10, features)<br/>y_train: label t+10<br/>X_test: (m, 10, features)<br/>y_test: label t+10"]
    
    C1 --> M1["🤖 Modèle XGBoost H1"]
    C2 --> M2["🤖 Modèle XGBoost H5"]
    C3 --> M3["🤖 Modèle XGBoost H10"]
    
    M1 --> R["✅ 3 modèles prédictifs<br/>indépendants"]
    M2 --> R
    M3 --> R
    
    style A fill:#fef3c7,stroke:#ca8a04,stroke-width:3px
    style B fill:#ddd6fe,stroke:#7c3aed,stroke-width:3px
    style C1 fill:#bbf7d0,stroke:#16a34a,stroke-width:2px
    style C2 fill:#bfdbfe,stroke:#2563eb,stroke-width:2px
    style C3 fill:#fed7aa,stroke:#ea580c,stroke-width:2px
    style M1 fill:#ccfbf1,stroke:#0d9488,stroke-width:2px
    style M2 fill:#ccfbf1,stroke:#0d9488,stroke-width:2px
    style M3 fill:#ccfbf1,stroke:#0d9488,stroke-width:2px
    style R fill:#fce7f3,stroke:#be185d,stroke-width:3px
```

---

## 📊 Performances

### Résultats du modèle XGBoost

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#10b981','secondaryColor':'#3b82f6'}}}%%
graph TD
    subgraph Metrics["📈 MÉTRIQUES DE PERFORMANCE"]
        A["🎯 Accuracy<br/>────────<br/>97%<br/>────────<br/>97 prédictions justes<br/>sur 100"]
        B["🎨 Precision<br/>────────<br/>96%<br/>────────<br/>96% des alertes<br/>sont réelles"]
        C["🔍 Recall<br/>────────<br/>94%<br/>────────<br/>94% des attaques<br/>détectées"]
        D["⚖️ F1-Score<br/>────────<br/>95%<br/>────────<br/>Équilibre<br/>optimal"]
    end
    
    style A fill:#bbf7d0,stroke:#15803d,stroke-width:3px,color:#000
    style B fill:#bfdbfe,stroke:#1e40af,stroke-width:3px,color:#000
    style C fill:#fed7aa,stroke:#c2410c,stroke-width:3px,color:#000
    style D fill:#e9d5ff,stroke:#7e22ce,stroke-width:3px,color:#000
```

### Visualisation des performances

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#06b6d4','secondaryColor':'#8b5cf6'}}}%%
%%{init: {'themeVariables': { 'xyChart': {'backgroundColor': '#f9fafb'}}}}%%
xychart-beta
    title "Performances du modèle"
    x-axis [Accuracy, Precision, Recall, F1-Score]
    y-axis "Score (%)" 0 --> 100
    bar [97, 96, 94, 95]
```

### Matrice de confusion conceptuelle

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#10b981','secondaryColor':'#ef4444'}}}%%
graph TD
    subgraph Confusion["Matrice de Confusion"]
        direction TB
        
        subgraph TruePos["✅ VRAIS POSITIFS<br/>────────<br/>94%<br/>────────<br/>Attaques correctement<br/>détectées"]
        end
        
        subgraph FalseNeg["❌ FAUX NÉGATIFS<br/>────────<br/>6%<br/>────────<br/>Attaques manquées"]
        end
        
        subgraph TrueNeg["✅ VRAIS NÉGATIFS<br/>────────<br/>Très élevé<br/>────────<br/>Trafic normal OK"]
        end
        
        subgraph FalsePos["⚠️ FAUX POSITIFS<br/>────────<br/>4%<br/>────────<br/>Fausses alertes"]
        end
    end
    
    style TruePos fill:#bbf7d0,stroke:#15803d,stroke-width:3px
    style FalseNeg fill:#fecaca,stroke:#b91c1c,stroke-width:3px
    style TrueNeg fill:#dbeafe,stroke:#1e40af,stroke-width:3px
    style FalsePos fill:#fed7aa,stroke:#c2410c,stroke-width:3px
```

 

### Comparaison avec solutions existantes

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#ef4444','secondaryColor':'#10b981'}}}%%
graph LR
    subgraph Classic["❌ Systèmes Classiques"]
        C1["Type:<br/>RÉACTIF"]
        C2["Anticipation:<br/>0 seconde"]
        C3["Efficacité:<br/>MOYENNE"]
    end
    
    subgraph Our["✅ Notre Système"]
        O1["Type:<br/>PRÉDICTIF"]
        O2["Anticipation:<br/>1-10 secondes"]
        O3["Efficacité:<br/>HAUTE (97%)"]
    end
    
    Classic -.->|vs| Our
    
    style Classic fill:#fee2e2,stroke:#b91c1c,stroke-width:3px
    style C1 fill:#fecaca,stroke:#b91c1c,stroke-width:2px
    style C2 fill:#fecaca,stroke:#b91c1c,stroke-width:2px
    style C3 fill:#fecaca,stroke:#b91c1c,stroke-width:2px
    
    style Our fill:#dcfce7,stroke:#15803d,stroke-width:3px
    style O1 fill:#bbf7d0,stroke:#15803d,stroke-width:2px
    style O2 fill:#bbf7d0,stroke:#15803d,stroke-width:2px
    style O3 fill:#bbf7d0,stroke:#15803d,stroke-width:2px
```

---

<div align="center">

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

**Datasets** : CICIDS2017 • CICDDOS2019

---

## 📫 Contact

**[Votre Nom]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](votre-lien)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](votre-lien)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:votre-email)

---

⭐ **Si ce projet vous intéresse, n'hésitez pas à lui donner une étoile !**

</div>
