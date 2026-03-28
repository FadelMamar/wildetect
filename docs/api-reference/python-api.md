# Python API Reference

!!! info "CLI-First Approach"
    WildDetect is designed to be used through its **command-line interface**. All functionality is exposed via the `wildetect`, `wildtrain`, and `wildata` CLI tools with YAML configuration files.

    **There is no public Python API** intended for direct programmatic use. The internal modules are implementation details and may change without notice.

## How to Use WildDetect

All operations are performed via the CLI and YAML config files:

### Detection

```bash
wildetect detection detect -c config/detection.yaml
```

See [WildDetect CLI Reference](wildetect-cli.md) for all commands.

### Data Management

```bash
wildata import-dataset -c configs/import-config-example.yaml
```

See [WilData CLI Reference](wildata-cli.md) for all commands.

### Model Training

```bash
wildtrain train detector -c configs/detection/yolo_configs/yolo.yaml
```

See [WildTrain CLI Reference](wildtrain-cli.md) for all commands.

---

## CLI Reference

| Package | CLI Reference | Config Reference |
|---------|--------------|------------------|
| WildDetect | [wildetect CLI](wildetect-cli.md) | [WildDetect Configs](../configs/wildetect/index.md) |
| WilData | [wildata CLI](wildata-cli.md) | [WilData Configs](../configs/wildata/index.md) |
| WildTrain | [wildtrain CLI](wildtrain-cli.md) | [WildTrain Configs](../configs/wildtrain/index.md) |

---

For step-by-step guides, see:

- [End-to-End Detection Tutorial](../tutorials/end-to-end-detection.md)
- [Dataset Preparation Tutorial](../tutorials/dataset-preparation.md)
- [Model Training Tutorial](../tutorials/model-training.md)
- [Census Campaign Tutorial](../tutorials/census-campaign.md)
