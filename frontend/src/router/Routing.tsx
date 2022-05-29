import { BrowserRouter, useRoutes } from 'react-router-dom'

import routes from '@router/routes'

const UsedRoutes = () => useRoutes(routes)

const Routing = () => (
  <BrowserRouter>
    <UsedRoutes />
  </BrowserRouter>
)
export default Routing
