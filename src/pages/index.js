import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="text--center">
          <img className="{styles.titleImage} animate__animated animate__zoomInDown" src={siteConfig.customFields.titleImage} alt={siteConfig.title} />
        </div>
        <p className="hero__subtitle animate__animated animate__fadeInUp animation-delay-100">{siteConfig.tagline}<br/>
          <small><small></small></small>
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg animate__animated animate__fadeInUp animation-delay-100"
            to="docs/getting-started/">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} Home`}
      description="A software package for analyzing Calabi-Yau manifolds.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
