import numpy as np
import pyircp


class CamPoseEstimator:
    def __init__(self, cam_intr, im_sz):
        kinvT = np.linalg.inv(cam_intr).T
        yg, xg = np.mgrid[0:im_sz[0], 0:im_sz[1]]
        self.r = np.c_[xg.flatten(), yg.flatten(),np.ones(xg.size)]@kinvT
        self.last_transform = np.eye(4)

        ircp_params = pyircp.ICPParams()
        ircp_params.maxIterations = 20
        ircp_params.maxNonIncreaseIterations = 3
        ircp_params.maxMatchingDistance = 0.5
        ircp_params.ransacIterations = 1000
        ircp_params.ransacMaxInlierDistance = 0.02
        ircp_params.seed = 0
        self.icp = pyircp.ICP(ircp_params)

        pass

    def depth2pts(self, depth_im):
        z = depth_im.flatten()
        msk = [z != 0]
        xyz = (self.r*z.reshape(-1,1))[msk]
        return xyz

    def estimate(self,depth_im,p_dst,n_dst):
        if len(p_dst)==0:
            return self.last_transform
        p_src = self.depth2pts(depth_im)

        res = self.icp.run(p_src, p_dst, n_dst, self.last_transform[:3,:3], self.last_transform[:3,3])

        self.last_transform[:3,:3]=res.rotation
        self.last_transform[:3, 3]=res.translation
        return self.last_transform



