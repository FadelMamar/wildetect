import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HeroSection() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started/installation">
            🚀 Get Started - 5min Quick Start
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({title, description, icon}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <span style={{fontSize: '4rem'}}>{icon}</span>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

const FeatureList = [
  {
    title: '1. Foundation: WilData',
    icon: '🗂️',
    description: (
      <>
        High-quality, version-controlled data infrastructure. Handles multi-format imports, 
        geospatial metadata extraction, and large-scale image tiling.
      </>
    ),
  },
  {
    title: '2. Intelligence: WildTrain',
    icon: '🎓',
    description: (
      <>
        Transform raw observations into specialized AI models. Flexible framework for 
        training state-of-the-art YOLO detectors and deep-learning classifiers.
      </>
    ),
  },
  {
    title: '3. Impact: WildDetect',
    icon: '🔍',
    description: (
      <>
        Deploy models in the field. Orchestrates census campaigns, processing 
        thousands of images to generate statistically sound population counts.
      </>
    ),
  },
];

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - AI for Wildlife Conservation`}
      description="Scalable and accurate wildlife monitoring using AI-driven aerial imagery analysis.">
      <HeroSection />
      <main>
        <section className={styles.mission}>
          <div className="container">
            <div className="row">
              <div className="col col--10 col--offset-1 text--center">
                <br />
                <Heading as="h2">The Mission</Heading>
                <p style={{fontSize: '1.2rem', lineHeight: '1.6'}}>
                  WildDetect is more than just a detection tool; it's a comprehensive <b>AI-driven ecosystem</b> 
                  designed to solve one of the most critical challenges in modern conservation: 
                  <b>scalable and accurate wildlife monitoring.</b>
                </p>
                <p style={{fontSize: '1.1rem'}}>
                  By automating the transition from raw aerial imagery to detailed census reports, 
                  WildDetect empowers researchers and conservationists to focus on protection and policy, 
                  rather than manual image scanning.
                </p>
                <hr />
              </div>
            </div>
          </div>
        </section>
        
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
