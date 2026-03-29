# 🗺️ Interactive Script Navigator

Use this interactive map to find the right script or CLI command for your task. **Click on any action** (the colored boxes) to jump directly to its documentation and configuration guide.

```mermaid
graph TB
    subgraph WilData ["📦 1. Data Preparation (WilData)"]
        direction TB
        WD1["Import Raw Data<br/>(COCO/YOLO/LS)"] --> WD2["Process & Tile"]
        WD2 --> WD3["Create ROI Dataset"]
        WD1 -.-> WD4["Visualize Labels"]
        
        click WD1 "../scripts/wildata/#import-dataset-examplebat" "Click to see Import scripts"
        click WD2 "../scripts/wildata/#bulk-import-datasetbat" "Click to see Bulk Processing"
        click WD3 "../scripts/wildata/#create-roi-datasetbat" "Click to see ROI scripts"
        click WD4 "../scripts/wildata/#visualize_databat" "Click to see Visualization"
    end

    subgraph WildTrain ["🎓 2. Model Training (WildTrain)"]
        direction TB
        WT1["Train YOLO Detector"] --> WT2["Register to MLflow"]
        WT3["Train Classifier"] --> WT2
        WT4["Run Full Pipeline"] --> WT2
        
        click WT1 "../scripts/wildtrain/#train_yolobat" "Click to see Detection training"
        click WT3 "../scripts/wildtrain/#train_classifierbat" "Click to see Classification training"
        click WT2 "../scripts/wildtrain/#register_modelbat" "Click to see Model registration"
        click WT4 "../scripts/wildtrain/#run_detection_pipelinebat" "Click to see Pipelines"
    end

    subgraph WildDetect ["🔍 3. Deployment & Analysis (WildDetect)"]
        direction TB
        D1["Run AI Detection"] --> D2["Run Census Campaign"]
        D2 --> D3["Generate Reports"]
        D4["Launch Dashboard UI"]
        D5["Visualize Predicts"]
        
        click D1 "../scripts/wildetect/#run_detectionbat" "Click to see Detection scripts"
        click D2 "../scripts/wildetect/#run_censusbat" "Click to see Census scripts"
        click D3 "../scripts/wildetect/#run_censusbat" "Click to see report generation"
        click D4 "../scripts/wildetect/#launch_uibat" "Click to see Dashboard"
        click D5 "../scripts/wildetect/#launch_fiftyonebat" "Click to see FiftyOne viewer"
    end

    WD3 --> WT3
    WD2 --> WT1
    WT2 --> D1
    
    %% Styling
    style WilData fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style WildTrain fill:#fff8e1,stroke:#ffc107,stroke-width:2px
    style WildDetect fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    
    style WD1 fill:#bbdefb,cursor:pointer
    style WD2 fill:#bbdefb,cursor:pointer
    style WD3 fill:#bbdefb,cursor:pointer
    style WD4 fill:#bbdefb,cursor:pointer
    
    style WT1 fill:#ffecb3,cursor:pointer
    style WT2 fill:#ffecb3,cursor:pointer
    style WT3 fill:#ffecb3,cursor:pointer
    style WT4 fill:#ffecb3,cursor:pointer
    
    style D1 fill:#c8e6c9,cursor:pointer
    style D2 fill:#c8e6c9,cursor:pointer
    style D4 fill:#c8e6c9,cursor:pointer
    style D5 fill:#c8e6c9,cursor:pointer
```

## How to use this Navigator
1. **Identify your current stage**: Are you preparing data, training models, or running analysis?
2. **Find your intent**: Each box represents a specific goal (e.g., "Import Raw Data").
3. **Click for details**: Clicking a box will take you to the exact section of the documentation describing the required `.bat` scripts and YAML configurations.

---

### Quick Access Summary

| Stage | Common Tasks | Entry Point |
| :--- | :--- | :--- |
| **Preparation** | Import, Tiling, ROI Extraction | [WilData Reference](scripts/wildata/) |
| **Training** | YOLO, Classification, Registration | [WildTrain Reference](scripts/wildtrain/) |
| **Inference** | Detection, Census, GIS Reports | [WildDetect Reference](scripts/wildetect/) |
| **Monitoring** | UI, Dashboard, MLflow | [Troubleshooting](troubleshooting.md) |
