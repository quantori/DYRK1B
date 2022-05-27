import { BrowserRouter, Route, Routes } from 'react-router-dom'

import Login from '@pages/Login'
import Settings from '@pages/Settings'
import The404 from '@pages/The404'

const Routing = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="login" element={<Login />} />
      <Route path="settings" element={<Settings />} />
      <Route path="*" element={<The404 />} />
    </Routes>
  </BrowserRouter>
)

export default Routing
