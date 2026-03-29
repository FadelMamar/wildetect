import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  defaultSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'script-navigator',
        'getting-started/quick-start',
        'getting-started/environment-setup',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/wildetect',
        'architecture/wildata',
        'architecture/wildtrain',
        'architecture/data-flow',
      ],
    },
    {
      type: 'category',
      label: 'Scripts Reference',
      items: [
        {
          type: 'category',
          label: 'WildDetect Scripts',
          items: [
            'scripts/wildetect/index',
            'scripts/wildetect/run_detection',
            'scripts/wildetect/run_census',
            'scripts/wildetect/launch_ui',
            'scripts/wildetect/launch_fiftyone',
            'scripts/wildetect/launch_labelstudio',
            'scripts/wildetect/launch_mlflow',
            'scripts/wildetect/launch_inference_server',
            'scripts/wildetect/register_model',
            'scripts/wildetect/extract_gps',
            'scripts/wildetect/profile_census',
            'scripts/wildetect/run_integration_tests',
            'scripts/wildetect/load_env',
          ],
        },
        {
          type: 'category',
          label: 'WilData Scripts',
          items: ['scripts/wildata/index'],
        },
        {
          type: 'category',
          label: 'WildTrain Scripts',
          items: ['scripts/wildtrain/index'],
        },
      ],
    },
    {
      type: 'category',
      label: 'Configuration Reference',
      items: [
        {
          type: 'category',
          label: 'WildDetect Configs',
          items: [
            'configs/wildetect/index',
            'configs/wildetect/detection',
            'configs/wildetect/census',
            'configs/wildetect/benchmark',
            'configs/wildetect/visualization',
            'configs/wildetect/extract-gps',
            'configs/wildetect/detector_registration',
            'configs/wildetect/config',
            'configs/wildetect/class_mapping',
          ],
        },
        {
          type: 'category',
          label: 'WilData Configs',
          items: [
            'configs/wildata/index',
            'configs/wildata/import-config',
            'configs/wildata/roi-config',
            'configs/wildata/gps-update-config',
          ],
        },
        {
          type: 'category',
          label: 'WildTrain Configs',
          items: [
            'configs/wildtrain/index',
            'configs/wildtrain/yolo-config-guide',
            'configs/wildtrain/classification-train',
            'configs/wildtrain/detection-train',
            'configs/wildtrain/registration',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/end-to-end-detection',
        'tutorials/dataset-preparation',
        'tutorials/model-training',
        'tutorials/census-campaign',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api-reference/wildetect-cli',
        'api-reference/wildata-cli',
        'api-reference/wildtrain-cli',
      ],
    },
    'troubleshooting',
  ],
};

export default sidebars;
