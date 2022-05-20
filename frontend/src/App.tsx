import './assets/index.css'

export default function App() {
  return (
    <h1>
      <span>My app </span>
      <code>{process.env.NODE_ENV} env</code>
    </h1>
  )
}
