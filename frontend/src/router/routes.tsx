import type { RouteObject } from 'react-router-dom'

import Login from '@pages/Login'
import Settings from '@pages/Settings'
import The404 from '@pages/The404'

const routes: RouteObject[] = [
  { path: '/', element: <Login /> },
  { path: 'login', element: <Login /> },
  { path: 'settings', element: <Settings /> },
  { path: '*', element: <The404 /> },
]

export default routes
