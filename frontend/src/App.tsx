import './assets/index.less'

const App = () => (
  <h1>
    <span>My app </span>
    <code>{process.env.NODE_ENV} env</code>
  </h1>
)

export default App
