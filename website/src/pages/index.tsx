import React, { JSX } from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import Button from '@site/src/components/ui/Button/Button';
import Card, { CardHeader, CardBody } from '@site/src/components/ui/Card/Card';
import styles from './index.module.css';

function HeroSection() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className="row">
          <div className={clsx('col col--7', styles.heroContent)}>
            <div className={styles.badge}>v3.0 - AI Ecosystem</div>
            <Heading as="h1" className={styles.heroTitle}>
              {siteConfig.title}
            </Heading>
            <p className={styles.heroSubtitle}>
              Architecting the future of <b>Wildlife Conservation</b> through
              scalable, intelligence-driven aerial imagery analysis.
            </p>
            <div className={styles.buttons}>
              <Button to="/docs/getting-started/installation" variant="primary">
                🚀 Get Started
              </Button>
              <Button to="/docs/architecture/overview" variant="ghost">
                Architecture →
              </Button>
            </div>
          </div>
          <div className={clsx('col col--5', styles.heroVisual)}>
            <div className={styles.visualStack}>
              <div className={styles.visualCard} style={{
                backgroundImage: 'url(img/hero.png)',
                backgroundSize: 'cover',
                backgroundPosition: 'center'
              }}>
                <div className={styles.visualOverlay}></div>
              </div>
              <div className={styles.visualGlass}></div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

const FeatureList = [
  {
    title: '1. WilData',
    icon: '🗂️',
    description: 'High-quality, version-controlled data infrastructure. Handles multi-format imports and large-scale tiling.',
    to: '/docs/architecture/wildata',
  },
  {
    title: '2. WildTrain',
    icon: '🎓',
    description: 'Transform raw observations into specialized AI models. Flexible framework for YOLO detectors.',
    to: '/docs/architecture/wildtrain',
  },
  {
    title: '3. WildDetect',
    icon: '🔍',
    description: 'Deploy models in the field. Orchestrates census campaigns and generates sound population counts.',
    to: '/docs/architecture/wildetect',
  },
];

const WorkflowStages = [
  {
    title: '1. Data Preparation',
    package: 'WilData',
    color: '#e3f2fd',
    steps: ['Raw Annotations (COCO/YOLO)', 'WilData Import & Tile', 'Processed Dataset'],
  },
  {
    title: '2. Model Training',
    package: 'WildTrain',
    color: '#fff3e0',
    steps: ['Model Architecture', 'Training Loop & HPO', 'MLflow Model Registry'],
  },
  {
    title: '3. Deployment',
    package: 'WildDetect',
    color: '#e8f5e9',
    steps: ['Aerial Image Stream', 'Multi-species Detections', 'Census Reports'],
  },
];

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Scalable and accurate wildlife monitoring using AI-driven aerial imagery analysis.">
      <HeroSection />
      <main>
        <section id="science" className={styles.publications}>
          <div className="container">
            <Heading as="h2" className="text--center">Scientific Impact</Heading>
            <div className={styles.pubGrid}>
              <Card className={styles.pubCard} shadow={true} to="https://nsojournals.onlinelibrary.wiley.com/doi/10.1002/wlb3.01523">
                <div className={styles.pubBadge}>Wildlife Biology (2026)</div>
                <Heading as="h3" className={styles.pubTitle}>
                  Evaluating machine learning models for multi-species wildlife detection and identification on remote sensed nadir imagery in South African savanna
                </Heading>
                <p className={styles.pubAuthors}>
                  Paul Allin, Fadel Seydou, Frans Radloff, Andrew Davies, Alison Leslie
                </p>
                <div className={styles.pubLink}>Read Publication →</div>
              </Card>

              <Card className={styles.pubCard} shadow={true} to="https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffvc">
                <div className={styles.pubBadge}>Dryad Repository</div>
                <Heading as="h3" className={styles.pubTitle}>
                  Dataset for: Evaluating machine learning models for multi-species wildlife detection and identification on remote sensed nadir imagery in South African savanna
                </Heading>
                <p className={styles.pubAuthors}>
                  Paul Allin, Fadel Seydou, Frans Radloff, Andrew Davies, Alison Leslie
                </p>
                <div className={styles.pubLink}>Access Data →</div>
              </Card>
            </div>
          </div>
        </section>

        <section id="contributions" className={styles.contributions}>
          <div className="container">
            <Heading as="h2" className="text--center">Ecosystem Contributions</Heading>
            <p className={clsx('text--center', styles.sectionSubtitle)}>
              Optimized and relabeled datasets to support the wider wildlife monitoring community.
            </p>
            <div className={styles.contributionsGrid}>
              <Card className={styles.pubCard} shadow={true} to="https://huggingface.co/datasets/fadel841/savmap">
                <div className={styles.pubBadge}>Hugging Face Datasets</div>
                <Heading as="h3" className={styles.pubTitle}>
                  SAVMAP: Ultra-high-resolution UAV imaging for rhinoceros conservation in Namibia
                </Heading>
                <p className={styles.pubAuthors}>
                  SAVMAP Project Team • Refined & Relabeled by WildDetect
                </p>
                <div className={styles.pubLink}>Explore Repository →</div>
              </Card>
            </div>
          </div>
        </section>

        <section id="mission" className={styles.mission}>
          <div className="container">
            <div className="row">
              <div className="col col--10 col--offset-1 text--center">
                <Heading as="h2">The Mission</Heading>
                <p className={styles.missionText}>
                  WildDetect is a comprehensive ecosystem designed for <b>scalable and accurate wildlife monitoring. </b>
                  By automating the transition from raw aerial imagery to detailed census reports,
                  we empower researchers to focus on policy and protection.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section id="ecosystem" className={styles.features}>
          <div className="container">
            <Heading as="h2" className="text--center">A Modular Ecosystem</Heading>
            <div className={styles.featuresGrid}>
              {FeatureList.map((props, idx) => (
                <Card key={idx} className={styles.featureCard} to={props.to}>
                  <CardHeader>
                    <span className={styles.featureIcon}>{props.icon}</span>
                  </CardHeader>
                  <CardBody>
                    <Heading as="h3">{props.title}</Heading>
                    <p>{props.description}</p>
                  </CardBody>
                </Card>
              ))}
            </div>
          </div>
        </section>
        <section id="workflow" className={styles.workflow}>
          <div className="container">
            <Heading as="h2" className="text--center">The Workflow</Heading>
            <div className={styles.pipeline}>
              {WorkflowStages.map((stage, idx) => (
                <React.Fragment key={idx}>
                  <div className={styles.pipelineNode} style={{ '--stage-color': stage.color } as any}>
                    <div className={styles.nodeHeader}>
                      <span className={styles.nodePackage}>{stage.package}</span>
                      <Heading as="h3">{stage.title}</Heading>
                    </div>
                    <ul className={styles.nodeSteps}>
                      {stage.steps.map((step, sIdx) => (
                        <li key={sIdx}>{step}</li>
                      ))}
                    </ul>
                  </div>
                  {idx < WorkflowStages.length - 1 && (
                    <div className={styles.pipelineConnector}>
                      <span className={styles.connectorArrow}>→</span>
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>
            <div className={clsx(styles.buttons, styles.workflowCTA)}>
              <Button to="/docs/getting-started/installation" variant="primary">
                🚀 Get Started with WildDetect
              </Button>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
