import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Fast algorithms',
    Svg: require('@site/static/img/2dpolys.svg').default,
    description: (
      <>
        CYTools includes fast algorithms to handle and triangulate reflexive
        polytopes and obtain the data of the associated Calabi-Yau hypersurfaces.
      </>
    ),
  },
  {
    title: 'Powerful and robust',
    Svg: require('@site/static/img/regular.svg').default,
    description: (
      <>
        CYTools can be used to explore the entirety of the Kreuzer-Skarke database,
        even at the largest Hodge numbers.
      </>
    ),
  },
  {
    title: 'Easy to use',
    Svg: require('@site/static/img/secfan.svg').default,
    description: (
      <>
        CYTools is easy to install and use. We provide extensive documentation and
        all the code is open-source.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
