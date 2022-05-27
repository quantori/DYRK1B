import { useNavigate } from 'react-router-dom'
import { Button } from 'antd'

const The404 = () => {
  const navigate = useNavigate()

  return (
    <div className="flex-grow flex flex-col justify-center items-center">
      <div>Page not found</div>
      <Button
        onClick={() => navigate(-1)}
        className="mt-2"
      >
        Go back
      </Button>
    </div>
  )
}

export default The404
