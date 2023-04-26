---
id: overview
title: Overview
slug: /getting-started/
---

import Link from '@docusaurus/Link';
import styles from './index.module.css';

CYTools is an open-source software package that relies on various other software packages built for Unix-like systems. It is primarily distributed in the form of a Docker image, so that it can easily be installed and used on Windows, macOS and most Linux distributions. However, for optimal performance we recommend a Linux installation since Docker uses the native kernel instead of using a virtual machine as in Windows and macOS.

We provide simple installation and usage instructions for the major operating systems. These are meant for most users, as they allow you to install and use CYTools without having to learn how Docker works or memorizing long commands.

<div className={styles.buttons}>
  <centered>
    <Link
    className="button button--primary"
    to="./linux">
    Linux
    </Link>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <Link
    className="button button--primary"
    to="./macos">
    macOS
    </Link>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <Link
    className="button button--primary"
    to="./windows">
    Windows
    </Link>
  </centered>
</div>
<br></br>

Once you are done with the installation, we encourage new users, especially those who are not familiar with Python, to follow the tutorial where we showcase the most important functionality, and give concrete examples of common computations one might be interested in.

<div className={styles.buttons}>
  <centered>
    <Link
    className="button button--primary"
    to="./tutorial">
    Tutorial
    </Link>
  </centered>
</div>
<br></br>

Finally, we provide more in-depth instructions for people who want to tinker with the Docker image, or want to have a special setup. We also list some performance tips for users who want to perform very extensive computations with a large number of polytopes or triangulations. For some projects it is necessary to be able to run CYTools on a cluster without admin privileges, so we have a separate guide for this purpose.

<div className={styles.buttons}>
  <centered>
    <Link
    className="button button--primary"
    to="./advanced">
    Advanced usage
    </Link>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <Link
    className="button button--primary"
    to="./cluster">
    Cluster installation
    </Link>
  </centered>
</div>
