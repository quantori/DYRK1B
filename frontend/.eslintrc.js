module.exports = {
  env: {
    browser: true,
    es2022: true,
  },
  extends: [
    'airbnb',
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 2022,
    sourceType: 'module',
  },
  plugins: [
    'react',
    '@typescript-eslint',
    'simple-import-sort',
  ],
  rules: {
    'max-len': ['warn', { code: 120, ignoreComments: true, ignoreTemplateLiterals: true }],
    semi: [
      'error',
      'never',
    ],
    'arrow-parens': [
      'error',
      'as-needed',
    ],
    'no-multiple-empty-lines': [
      'warn',
      {
        max: 1,
        maxBOF: 0,
      },
    ],
    'simple-import-sort/imports': [
      'warn',
      {
        groups: [
          // Style imports
          ['^.+\\.(s?css|less)$'],
          // react related packages, other packages
          ['^react', '(\\w-/)*'],
          // Side effect imports, Alias, Relative
          [
            '^\\u0000',
            '^@assets',
            '^@pages',
            '^@store',
            '^@router',
            '^@',
            '^\\.',
          ],
        ],
      },
    ],
    'import/extensions': [
      'error',
      'ignorePackages',
      {
        ts: 'never',
        tsx: 'never',
      },
    ],
    'import/no-extraneous-dependencies': ['error', { devDependencies: ['**/*.test.ts', '**/*.test.tsx'] }],
    'lines-between-class-members': [
      'error',
      'always',
      { exceptAfterSingleLine: true },
    ],
    'no-useless-constructor': 'off',
    '@typescript-eslint/no-useless-constructor': ['error'],
    'no-shadow': 'off',
    '@typescript-eslint/no-shadow': ['error'],
    'function-call-argument-newline': ['error', 'consistent'],
    'function-paren-newline': ['error', 'multiline-arguments'],
    'padding-line-between-statements': [
      'warn',
      {
        blankLine: 'always',
        prev: '*',
        next: 'block-like',
      },
    ],
    'react/function-component-definition': [
      'warn',
      { namedComponents: 'arrow-function' },
    ],
    'react/jsx-filename-extension': [
      'warn',
      {
        extensions: ['.tsx'],
      },
    ],
    'react/jsx-one-expression-per-line': 'off',
    'react/prop-types': 'off',
    // unnecessary rules https://reactjs.org/blog/2020/09/22/introducing-the-new-jsx-transform.html#eslint
    'react/jsx-uses-react': 'off',
    'react/react-in-jsx-scope': 'off',
    'react/require-default-props': 'off',
    'jsx-a11y/click-events-have-key-events': 'off',
    'jsx-a11y/interactive-supports-focus': 'off',
    '@typescript-eslint/no-non-null-assertion': 'off',
    'class-methods-use-this': 'off',
    'no-underscore-dangle': ['error', { allowAfterThis: true }],
    camelcase: ['error', { allow: ['given_name', 'family_name'] }],
    '@typescript-eslint/type-annotation-spacing': ['warn',
      {
        after: true,
      },
    ],
    '@typescript-eslint/member-delimiter-style': [
      'warn',
      {
        multiline: {
          delimiter: 'none',
        },
        singleline: {
          delimiter: 'comma',
        },
      },
    ],
  },
  settings: {
    'import/resolver': {
      typescript: {},
    },
  },
}
