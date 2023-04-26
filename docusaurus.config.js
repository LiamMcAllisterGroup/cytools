// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

const remarkMath = require('remark-math')
const rehypeKatex = require('rehype-katex')

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'CYTools',
  tagline: 'A software package for analyzing Calabi-Yau manifolds',
  url: 'https://cy.tools',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Liam McAllister Group', // Usually your GitHub org/user name.
  projectName: 'cytools', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeKatex, {strict: false}]],
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'CYTools',
        logo: {
          alt: 'CYTools',
          src: 'img/logo.svg',
        },
        items: [
          {to: 'docs/getting-started/', label: 'Get Started', position: 'left'},
          {
            to: 'docs/documentation/',
            activeBasePath: 'docs/documentation/',
            label: 'Docs',
            position: 'left',
          },
          {
            href: 'https://github.com/LiamMcAllisterGroup/cytools',
            label: 'GitHub',
            position: 'left',
          },
          {to: 'about/', label: 'About', position: 'left'},
          {
            href: 'https://liammcallistergroup.com',
            label: 'Liam McAllister Group',
            position: 'right',
          },
        ],
      },
      algolia: {
        appId: 'OWJRSPYL3L',
        apiKey: '7c68c6dd741669b52f19b4079c720917',
        indexName: 'cytools',
        contextualSearch: true,
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Getting Started',
            items: [
              {
                label: 'Installation',
                to: 'docs/getting-started/',
              },
              {
                label: 'Tutorial',
                to: 'docs/getting-started/tutorial',
              },
            ],
          },
          {
            title: 'Docs',
            items: [
              {
                label: 'Documentation',
                to: 'docs/documentation/',
              },
              {
                label: 'License',
                to: 'docs/documentation/license',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'About CYTools',
                to: 'about',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/LiamMcAllisterGroup/cytools',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Liam McAllister Group.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        //theme: require('prism-react-renderer/themes/oceanicNext'),
      },
    }),
    scripts: ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML', '/js/mathjax-config.js'],
    customFields: {
      titleImage: 'img/titleimage.svg',
    }
};

module.exports = config;
