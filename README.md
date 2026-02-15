# Forecasting DDoS Attacks 1 to 10 Seconds Before Occurrence

<div align="center">

 
We propose an AI-powered predictive system designed to continuously analyze network traffic and identify early precursor patterns of a DDoS attack before full-scale impact.

The objective is to transition from traditional reactive mitigation strategies to proactive, real-time threat anticipation using machine learning.


 

</div>

---

## The Challenge

Distributed Denial of Service (DDoS) attacks aim to overwhelm target servers by generating a massive volume of malicious requests from a network of compromised machines (botnets).  

These attacks result in service outages, preventing legitimate users from accessing essential resources.

 [![DDoS Attack Explanation](https://img.shields.io/badge/DDoS-Explanation-F38020?style=flat-square&logo=cloudflare&logoColor=white)](https://www.cloudflare.com/fr-fr/learning/ddos/what-is-a-ddos-attack/)



 

---

## Core Concept

DDoS attacks do not reach their peak instantaneously. Instead, they exhibit characteristic signatures in the seconds preceding the traffic flood, including:

- A progressive surge in traffic volume  
- Shifting connection patterns  
- Increasing request homogenization  

These early indicators provide a critical opportunity for anticipation.

---

##  Architecture

 
![Schéma d'architecture](Diagramme.svg)
## Our Methodology

### Time-Series Analysis
Network traffic is treated as a continuous time series rather than a collection of isolated events, allowing the system to capture temporal dependencies and evolving patterns.

### Proactive Forecasting
The model analyzes traffic data from the previous **30 seconds** to predict the network state **1, 5, and 10 seconds** into the future.

### Paradigm Shift
By reframing intrusion detection as a **temporal prediction problem**, we move the defense line forward—enabling mitigation **before** the attack fully materializes.


 
### How Do We Create Temporal Sequences?

This is the core of our approach.

Instead of analyzing each second of network traffic in isolation, we construct **temporal sequences** using observation windows that capture the dynamic evolution of traffic over time.

---

## Dataset Requirements

Before generating temporal sequences, the dataset must satisfy two critical conditions:

### 1. Chronological Ordering

The dataset must be strictly ordered by time.

- Rows must follow a consistent temporal sequence.
- Any shuffled or unordered data will break temporal consistency.
- Time continuity is essential for learning meaningful sequential patterns.

### 2. Uniform Temporal Granularity

All rows must share the same time granularity.

In our case:

- After feature selection and data cleaning,
- The dataset was aggregated **per second**.

This means:

> Each row represents exactly **one second of network traffic information**.

Uniform granularity ensures:
- Temporal consistency
- Stable window construction
- Reliable forecasting behavior

---

#### Sliding Window Principle

We generate fixed-length time windows that move forward progressively.

- Each window spans a predefined duration (e.g., **30 seconds**).
- The data within this window forms a **single input sequence** for the model.
- The window then shifts forward by a fixed step (e.g., **1 second**), producing a new sequence.

This creates overlapping sequences that preserve temporal continuity.

---
 

 

If:

- Window size = 30 seconds  
- Step size = 1 second  
- Dataset granularity = 1 row per second  

Then:

- Sequence 1 → seconds 1–30  
- Sequence 2 → seconds 2–31  
- Sequence 3 → seconds 3–32  
- ...

This produces a continuously updating predictive pipeline.


```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#5D8571','secondaryColor':'#94B5A3', 'fontSize':'20px', 'primaryTextColor':'#ffffff', 'secondaryTextColor':'#ffffff'}}}%%
graph LR
    subgraph Window1["Window 1"]
        direction LR
        A0["t0"] --> A1["t1"] --> A2["t2"] -.-> A3["..."] -.-> A28["t28"] --> A29["t29"]
    end
    
    Window1 -.->|Prediction| P1["Label t34"]
    
    style Window1 fill:#e9ecef,stroke:#495057,stroke-width:2px, 'fontSize':'20px
    style P1 fill:#94B5A3,stroke:#94B5A3,stroke-width:2px,color:#ffffff, 'fontSize':'20px
    style A0 color:#ffffff
    style A1 color:#ffffff
    style A2 color:#ffffff
    style A3 color:#ffffff
    style A28 color:#ffffff
    style A29 color:#ffffff
```
**Calculation**:
- Input: `data[0:30]` → data from times 0, 1, 2, 3, ..., 27, 28, 29
- Output: `label[0 + 30 + 5 - 1]` = `label[34]`


**For the second sequence (position i=1)**:
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#5D8571','secondaryColor':'#94B5A3', 'fontSize':'20px', 'primaryTextColor':'#ffffff', 'secondaryTextColor':'#ffffff'}}}%%
graph LR
    subgraph Window2["Window 2 (shifted +1)"]
        direction LR
        B1[t1] --> B2[t2] --> B3[t3] -.-> B4[...] -.-> B29[t29] --> B30[t30]
    end
    
    Window2 -.->|Prediction| P2[Label t35]
    
    style Window2 fill:#e9ecef,stroke:#495057,stroke-width:2px, 'fontSize':'20px
    style P2 fill:#94B5A3,stroke:#94B5A3,stroke-width:2px,color:#ffffff, 'fontSize':'20px
    style B1 font-size:10px
    style B2 font-size:10px
    style B3 font-size:10px
    style B4 font-size:10px
    style B29 font-size:10px
    style B30 font-size:10px
```

 
**Calculation**:
- Input: `data[1:31]` → data from times 1, 2, 3, 4, ..., 28, 29, 30
- Output: `label[1 + 30 + 5 - 1]` = `label[35]`


**For the third sequence (position i=2)**:
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#5D8571','secondaryColor':'#94B5A3', 'fontSize':'20px', 'primaryTextColor':'#ffffff', 'secondaryTextColor':'#ffffff'}}}%%
graph LR
    subgraph Window3["Window 3 (shifted +2)"]
        direction LR
        C2[t2] --> C3[t3] --> C4[t4] -.-> C5[...] -.-> C30[t30] --> C31[t31]
    end
    
    Window3 -.->|Prediction| P3[Label t36]
    
    style Window3 fill:#e9ecef,stroke:#495057,stroke-width:2px, 'fontSize':'20px
    style P3 fill:#94B5A3,stroke:#94B5A3,stroke-width:2px,color:#ffffff, 'fontSize':'20px
    style C2 font-size:10px
    style C3 font-size:10px
    style C4 font-size:10px
    style C5 font-size:10px
    style C30 font-size:10px
    style C31 font-size:10px
```

 
**Calculation**:
- Input: `data[2:32]` → data from times 2, 3, 4, 5, ..., 29, 30, 31
- Output: `label[2 + 30 + 5 - 1]` = `label[36]`
 

 

 
 
#### General formula


For any sequence at position i:

**Calculation**:
- Input: `data[i:i+seq_length]` → data from times i, i+1, i+2, ..., i+(seq_length-1)
- Output: `label[i + seq_length + horizon - 1]`

 

 
 
---

<div align="center">

 

**Datasets** :  
[![Dataset IDS-2017](https://img.shields.io/badge/Dataset-IDS2017-blue?style=for-the-badge)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Dataset CICDDoS2019](https://img.shields.io/badge/Dataset-CICDDoS2019-red?style=for-the-badge)](https://www.unb.ca/cic/datasets/ddos-2019.html)


---

##  Contact

**[IDER TILLELI]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tilleli-ider-b0ba18221/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:idertilleli@gmail.com)

---

 

</div>
