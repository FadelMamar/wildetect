import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/wildetect/__docusaurus/debug',
    component: ComponentCreator('/wildetect/__docusaurus/debug', '1cc'),
    exact: true
  },
  {
    path: '/wildetect/__docusaurus/debug/config',
    component: ComponentCreator('/wildetect/__docusaurus/debug/config', '9dd'),
    exact: true
  },
  {
    path: '/wildetect/__docusaurus/debug/content',
    component: ComponentCreator('/wildetect/__docusaurus/debug/content', '627'),
    exact: true
  },
  {
    path: '/wildetect/__docusaurus/debug/globalData',
    component: ComponentCreator('/wildetect/__docusaurus/debug/globalData', 'c00'),
    exact: true
  },
  {
    path: '/wildetect/__docusaurus/debug/metadata',
    component: ComponentCreator('/wildetect/__docusaurus/debug/metadata', 'ca6'),
    exact: true
  },
  {
    path: '/wildetect/__docusaurus/debug/registry',
    component: ComponentCreator('/wildetect/__docusaurus/debug/registry', 'c11'),
    exact: true
  },
  {
    path: '/wildetect/__docusaurus/debug/routes',
    component: ComponentCreator('/wildetect/__docusaurus/debug/routes', '408'),
    exact: true
  },
  {
    path: '/wildetect/markdown-page',
    component: ComponentCreator('/wildetect/markdown-page', '719'),
    exact: true
  },
  {
    path: '/wildetect/docs',
    component: ComponentCreator('/wildetect/docs', '616'),
    routes: [
      {
        path: '/wildetect/docs',
        component: ComponentCreator('/wildetect/docs', '9bb'),
        routes: [
          {
            path: '/wildetect/docs',
            component: ComponentCreator('/wildetect/docs', 'ce4'),
            routes: [
              {
                path: '/wildetect/docs',
                component: ComponentCreator('/wildetect/docs', '14a'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/api-reference/wildata-cli',
                component: ComponentCreator('/wildetect/docs/api-reference/wildata-cli', '88a'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/api-reference/wildetect-cli',
                component: ComponentCreator('/wildetect/docs/api-reference/wildetect-cli', '60e'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/api-reference/wildtrain-cli',
                component: ComponentCreator('/wildetect/docs/api-reference/wildtrain-cli', 'c7a'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/architecture/data-flow',
                component: ComponentCreator('/wildetect/docs/architecture/data-flow', '4f1'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/architecture/overview',
                component: ComponentCreator('/wildetect/docs/architecture/overview', '6d8'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/architecture/wildata',
                component: ComponentCreator('/wildetect/docs/architecture/wildata', '5bb'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/architecture/wildetect',
                component: ComponentCreator('/wildetect/docs/architecture/wildetect', '211'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/architecture/wildtrain',
                component: ComponentCreator('/wildetect/docs/architecture/wildtrain', '8e4'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildata',
                component: ComponentCreator('/wildetect/docs/configs/wildata', '2ba'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildata/gps-update-config',
                component: ComponentCreator('/wildetect/docs/configs/wildata/gps-update-config', '51e'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildata/import-config',
                component: ComponentCreator('/wildetect/docs/configs/wildata/import-config', 'b5f'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildata/roi-config',
                component: ComponentCreator('/wildetect/docs/configs/wildata/roi-config', 'b6d'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect',
                component: ComponentCreator('/wildetect/docs/configs/wildetect', 'd2e'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/benchmark',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/benchmark', '981'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/census',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/census', '665'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/class_mapping',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/class_mapping', 'a1f'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/config',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/config', '9c6'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/detection',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/detection', '4c6'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/detector_registration',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/detector_registration', '0e9'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/extract-gps',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/extract-gps', '3b5'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildetect/visualization',
                component: ComponentCreator('/wildetect/docs/configs/wildetect/visualization', 'de9'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildtrain',
                component: ComponentCreator('/wildetect/docs/configs/wildtrain', 'a84'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildtrain/classification-train',
                component: ComponentCreator('/wildetect/docs/configs/wildtrain/classification-train', '066'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildtrain/detection-train',
                component: ComponentCreator('/wildetect/docs/configs/wildtrain/detection-train', '416'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildtrain/registration',
                component: ComponentCreator('/wildetect/docs/configs/wildtrain/registration', 'fc9'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/configs/wildtrain/yolo-config-guide',
                component: ComponentCreator('/wildetect/docs/configs/wildtrain/yolo-config-guide', 'faf'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/getting-started/environment-setup',
                component: ComponentCreator('/wildetect/docs/getting-started/environment-setup', '41c'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/getting-started/installation',
                component: ComponentCreator('/wildetect/docs/getting-started/installation', 'cfe'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/getting-started/quick-start',
                component: ComponentCreator('/wildetect/docs/getting-started/quick-start', '8ca'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/script-navigator',
                component: ComponentCreator('/wildetect/docs/script-navigator', '225'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildata',
                component: ComponentCreator('/wildetect/docs/scripts/wildata', '02a'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect', 'b3f'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/extract_gps',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/extract_gps', '51b'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/launch_fiftyone',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/launch_fiftyone', '0e2'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/launch_inference_server',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/launch_inference_server', '672'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/launch_labelstudio',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/launch_labelstudio', 'e65'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/launch_mlflow',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/launch_mlflow', '3e7'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/launch_ui',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/launch_ui', '375'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/load_env',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/load_env', '3b3'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/profile_census',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/profile_census', '21a'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/register_model',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/register_model', 'c85'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/run_census',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/run_census', '098'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/run_detection',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/run_detection', '30c'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildetect/run_integration_tests',
                component: ComponentCreator('/wildetect/docs/scripts/wildetect/run_integration_tests', 'd12'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/scripts/wildtrain',
                component: ComponentCreator('/wildetect/docs/scripts/wildtrain', '1dc'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/troubleshooting',
                component: ComponentCreator('/wildetect/docs/troubleshooting', 'ced'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/tutorials/census-campaign',
                component: ComponentCreator('/wildetect/docs/tutorials/census-campaign', '09a'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/tutorials/dataset-preparation',
                component: ComponentCreator('/wildetect/docs/tutorials/dataset-preparation', '532'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/tutorials/end-to-end-detection',
                component: ComponentCreator('/wildetect/docs/tutorials/end-to-end-detection', '99c'),
                exact: true,
                sidebar: "defaultSidebar"
              },
              {
                path: '/wildetect/docs/tutorials/model-training',
                component: ComponentCreator('/wildetect/docs/tutorials/model-training', 'ec1'),
                exact: true,
                sidebar: "defaultSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/wildetect/',
    component: ComponentCreator('/wildetect/', '6cc'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
