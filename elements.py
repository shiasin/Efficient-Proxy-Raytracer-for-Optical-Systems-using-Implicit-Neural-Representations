import torch
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F

 
class Sensor:
    """
        Can represent both sensor or a grid source light
    """
    def __init__(self, sensor_dict, rand=False,normalize = False):
        self.load_sensor(sensor_dict)
        self.normalize = normalize
        self.pixels = self.create_pixel_grid(rand)

    def load_sensor(self, sensor_dict):

        self.sensor_size = torch.tensor(sensor_dict["sensor_size"], dtype=torch.float64)
        self.sensor_res = torch.tensor(sensor_dict["sensor_res"], dtype=torch.float64)

        self.pixel_pitch_x = self.sensor_size[1] / self.sensor_res[1]
        self.pixel_pitch_y = self.sensor_size[0] / self.sensor_res[0]
        self.center = torch.tensor(sensor_dict.get("center", [0, 0, 0]), dtype=torch.float64)
        self.raysperpix = torch.tensor(sensor_dict['samples_per_pixel'])


    def create_pixel_grid(self,rand):

        num_x, num_y = self.sensor_res[1].item(), self.sensor_res[0].item()

        if not rand:
            pixels = self._define_pixels(
                center=self.center[:2],
                height=self.sensor_size[0].item(),
                width=self.sensor_size[1].item(),
                pixel_pitch_x=self.pixel_pitch_x,
                pixel_pitch_y=self.pixel_pitch_y
            )
        else:
            pixels = self._define_random_pixels(
                center=self.center[:2],
                height=self.sensor_size[0].item(),
                width=self.sensor_size[1].item(),
                pixel_pitch_x=self.pixel_pitch_x,
                pixel_pitch_y=self.pixel_pitch_y
            )
        # pixels = torch.tensor([[0.0, 0.0, 0.0]],  dtype=torch.float64)
        # pixels = pixels[13:14, :]

        return pixels.unsqueeze(1)


    def _define_pixels(self, center, height, width, pixel_pitch_x, pixel_pitch_y):

        num_x = int(self.sensor_res[1].item())
        num_y = int(self.sensor_res[0].item())

        x_coords = torch.linspace(center[0] - width / 2, center[0] + width / 2, num_x)
        y_coords = torch.linspace(center[1] - height / 2, center[1] + height / 2, num_y)
        xv, yv = torch.meshgrid(x_coords, y_coords, indexing='ij')


        denom_x = (xv.max() - xv.min()) if (xv.max() - xv.min()) > 0 else 1
        denom_y = (yv.max() - yv.min()) if (yv.max() - yv.min()) > 0 else 1
        xv = 2 * (xv - xv.min()) / denom_x - 1
        yv = 2 * (yv - yv.min()) / denom_y - 1

        if not self.normalize:
            xv = xv * (width / 2 - (pixel_pitch_x / width * (width / 2)) / 2)
            yv = yv * (height / 2 - (pixel_pitch_y / height * (height / 2)) / 2)

        starting_points = torch.stack([xv.flatten(), yv.flatten(), torch.zeros_like(yv.flatten())], dim=-1)
        return starting_points
    
    def _define_random_pixels(self, center, height, width, pixel_pitch_x, pixel_pitch_y):
        """
            A source light with random structure for more accurate results and to remove limitations to a source light at specific displacement distance.(not used )      
            """
        num_x = int(self.sensor_res[1].item())
        num_y = int(self.sensor_res[0].item())

        x_coords = torch.rand(num_x)*((center[0] + width / 2) - (center[0] - width / 2)) +  (center[0] - width / 2)
        y_coords = torch.rand(num_y)*((center[0] + height / 2) - (center[0] - height / 2)) +  (center[0] - height / 2)

        xv, yv = torch.meshgrid(x_coords, y_coords, indexing='ij')


        denom_x = (xv.max() - xv.min()) if (xv.max() - xv.min()) > 0 else 1
        denom_y = (yv.max() - yv.min()) if (yv.max() - yv.min()) > 0 else 1
        xv = 2 * (xv - xv.min()) / denom_x - 1
        yv = 2 * (yv - yv.min()) / denom_y - 1

        if not self.normalize:
            xv = xv * (width / 2 - (pixel_pitch_x / width * (width / 2)) / 2)
            yv = yv * (height / 2 - (pixel_pitch_y / height * (height / 2)) / 2)

        starting_points = torch.stack([xv.flatten(), yv.flatten(), torch.zeros_like(yv.flatten())], dim=-1)
        return starting_points

    def emitter(self, points):
        num_pixels = self.pixels.shape[0]
        num_points = points.shape[0]

        pixels_repeated = self.pixels.repeat(num_points, 1, 1)  # (n * m, 1, 3)

        points_repeated = points.repeat_interleave(num_pixels, dim=0)  # (n * m, 1, 3)
        directions = points_repeated - pixels_repeated  # (n * m, 1, 3)
        directions = 1* directions / directions.norm(dim=-1, keepdim=True)

        rays = torch.cat([pixels_repeated, directions], dim=1)  # (n * m, 2, 3)
        rays_emit = torch.cat([rays, rays, torch.zeros((rays.shape[0], 2, 1))], dim=-1) 

       
        return rays_emit, points_repeated

    def __repr__(self):
        return f"Sensor(size={self.sensor_size.tolist()}, resolution={self.sensor_res.tolist()}, center={self.center.tolist()})"

    def __call__(self, points):
        return self.emitter(points)


class Surface:
    """
    Base class for all surfaces.
    """
    def __init__(self):
        pass

    @property
    def normal(self, points):

        raise NotImplementedError("normal method must be overridden.")

    def intersection(self, rays):
        raise NotImplementedError("intersection method must be overridden.")


    def refract(self, vector, points, n1, n2, error=0.01):
        """
        Refract an incoming ray using the plane's normal vectors.

        Args:
            vector (torch.tensor): Incoming ray. (m, 2, 7).
            points (torch.tensor): intersection points (num_points, 1, 3).
            n1 (float): Refractive index of the incoming medium.
            n2 (float): Refractive index of the outgoing medium.
            error (float): Desired error.

        Returns:
            torch.tensor: Refracted ray with shape [m, 2, 3].
        """
        # Compute normal vectors at the given points
        normvector = self.normal(points)  # (num_points, 2, 3)
        # print(normvector.shape)

        if len(vector.shape) == 2:
            vector = vector.unsqueeze(0)
        if len(normvector.shape) == 2:
            normvector = normvector.unsqueeze(0)
        mu = n1 / n2
        div = normvector[:, 1, 0] ** 2 + normvector[:, 1, 1] ** 2 + normvector[:, 1, 2] ** 2
        a = mu * (vector[:, 1, 0] * normvector[:, 1, 0] +
                  vector[:, 1, 1] * normvector[:, 1, 1] +
                  vector[:, 1, 2] * normvector[:, 1, 2]) / div
        # print('---------------------')
        b = (mu ** 2 - 1) / div
        to = -b * 0.5 / a
        num = 0
        eps = torch.ones(vector.shape[0], device=vector.device) * error * 2
        while len(eps[eps > error]) > 0:
            num += 1
            oldto = to
            v = to ** 2 + 2 * a * to + b
            deltav = 2 * (to + a)
            to = to - v / deltav
            eps = abs(oldto - to)

        output = torch.zeros_like(vector)

        output[:, 0, 0] = normvector[:, 0, 0]
        output[:, 0, 1] = normvector[:, 0, 1]
        output[:, 0, 2] = normvector[:, 0, 2]
        output[:, 1, 0] = mu * vector[:, 1, 0] + to * normvector[:, 1, 0]
        output[:, 1, 1] = mu * vector[:, 1, 1] + to * normvector[:, 1, 1]
        output[:, 1, 2] = mu * vector[:, 1, 2] + to * normvector[:, 1, 2]
        output[:, 0, 3:] = vector[:, 0,3:]
        output[:, 1, 3:] = vector[:, 1,3:]

        return output

#It should be updated for the plano surface of cylindreical lens
class PlaneSurface(Surface):
    """
    A class representing a plane surface with a defined radius.
    """
    def __init__(self, center, normal_vector, radius):

        super().__init__()
        self.center = center  # (1, 1, 3)
        self.normal_vector = normal_vector / torch.norm(normal_vector, dim=-1, keepdim=True)  # Normalize the normal vector
        self.radius = radius  # Radius of the plane
        self.pos = center

    def normal(self, points):

        num_points = points.shape[0]

        normal_vector_expanded = self.normal_vector.repeat(num_points,1 ,1)  # (num_points, 3)
        normals = torch.cat((points, normal_vector_expanded), dim=1)  # (num_points, 2, 3)
        # print(normals)
        return normals

    # N.(p-p0)= 0
    def intersection(self, rays):
  
        ray_origins = rays[:, 0, :3]
        ray_directions = rays[:, 1, :3]
        # Calculate the denominator: ray_directions . normal_vector
        denom = torch.sum(ray_directions * self.normal_vector.squeeze(1), dim=1)  # (num_rays,)
        valid_mask = denom != 0
        t = torch.zeros_like(denom)
        t[valid_mask] = torch.sum(
            (self.center.squeeze(1) - ray_origins[valid_mask]) * self.normal_vector.squeeze(1),
            dim=1
        ) / denom[valid_mask]  # (num_rays,)

        intersection_points = torch.zeros_like(ray_origins)
        intersection_points[valid_mask] = ray_origins[valid_mask] + t[valid_mask].unsqueeze(1) * ray_directions[valid_mask]

        distances = torch.norm(intersection_points[:,:2] - self.center.squeeze(1)[:,:2], dim=1)  # (num_rays,)

        radius_mask = distances <= self.radius

        final_mask = valid_mask & radius_mask
    
        intersection_points[~final_mask] = float('nan')
        intersected_rays = rays.clone()
        intersected_rays[~final_mask] =float('nan')

        return intersection_points.unsqueeze(1), intersected_rays, final_mask


class SphereSurface(Surface):

    def __init__(self, center, radius, cut_radius):
        """
        Args:
            center (torch.Tensor): The center point of the sphere (1, 1, 3).
            radius (float): The radius of the sphere.
            cut_radius (float): The cut-off radius for cases that a part of sphere is needed.
        """
        super().__init__()
        self.center = center  #(1, 1, 3)  [2,2,100]
        self.radius = radius  #Radius of the sphere
        self.cut_radius = cut_radius  #Cut radius in the xy-plane
        self.pos = center - torch.tensor([0,0,radius])
        # print(self.pos)
        # exit()
    def normal(self, points):
        return torch.cat((points[...,:3], (points[...,:3] - self.center)/ torch.norm((points[...,:3] - self.center), dim=-1, keepdim=True)), dim=1)

    # simple intersection  with mathematical formula
    def intersection_w(self, rays):

        ray_origins = rays[:, 0, :3]  # (num_rays, 3)
        ray_directions = rays[:, 1, :3]  # (num_rays, 3)

        # Compute quadratic coefficients
        oc = ray_origins - self.center.squeeze(1)
        A = torch.sum(ray_directions ** 2, dim=-1)
        B = 2.0 * torch.sum(ray_directions * oc, dim=-1)
        C = torch.sum(oc * oc, dim=-1) - self.radius ** 2

        # Compute discriminant and check for valid intersections
        discriminant = B**2 - 4*A*C
        valid_mask = discriminant >= 0
        # print('log')
        # print(valid_mask.shape)
        
        # Compute roots where valid
        sqrt_discriminant = torch.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)
        

        valid_t1, valid_t2 = ((t1 > 0 )& valid_mask), ((t2 > 0) & valid_mask)
        print(valid_t1.shape, valid_t2.shape, t1.shape)
        t_final = torch.where(valid_t1 & valid_t2, torch.min(t1, t2), torch.where(valid_t1, t1, torch.where(valid_t2, t2, float('nan'))))
        valid_mask = ~torch.isnan(t_final)

        # Compute intersection points
        intersection_points = torch.full_like(ray_origins, float('nan'))
   
        intersection_points[valid_mask] = ray_origins[valid_mask] + t_final[valid_mask].unsqueeze(1) * ray_directions[valid_mask]

        distances_xy = (intersection_points[:,0 ]-self.pos.squeeze()[0]) ** 2 + (intersection_points[:, 1]-self.pos.squeeze()[1]) ** 2
        radius_mask = distances_xy <= (self.cut_radius**2)

        
        final_mask = ~valid_mask | ~radius_mask

        rays_out = rays.clone()
        rays_out[final_mask] = float('nan')

        return intersection_points.unsqueeze(1), rays_out, valid_mask

    #iterative newton method
    def intersection(self, rays, number_of_steps=200, error_threshold=1e-8):
        ray_origins = rays[:, 0, :3] 
        ray_directions = F.normalize(rays[:, 1, :3], dim=-1)  # (N, 3)
        num_rays = ray_origins.shape[0]

   
        dist = ((self.pos[...,2] - ray_origins[...,2]) / ray_directions[...,2]).squeeze(0)
        def compute_error_and_derivative(d):
            p = ray_origins + d.unsqueeze(-1) * ray_directions 
            diff = p[...,:3] - self.center.squeeze(0) 
            error = (diff ** 2).sum(dim=-1) - self.radius ** 2  
            d_error = 2 * (diff * ray_directions).sum(dim=-1) 
            return error, d_error

        t = tqdm(range(number_of_steps), leave=False, dynamic_ncols=True)

        active_mask = torch.ones(num_rays, dtype=torch.bool, device=ray_origins.device)

        for step in t:
            error, d_error = compute_error_and_derivative(dist)
            # safe = d_error.abs() > 1e-8

            # d_next = d - f / f'
            delta = error / (d_error+1e-8)
            # delta = torch.where(safe, error / d_error, torch.zeros_like(d_error))
            dist = dist - delta
            dist = dist.clamp(min=0) 

            active_mask &= error.abs() > error_threshold
            if not active_mask.any():
                break

            t.set_description(f"Newton sphere intersection loss: {error[active_mask].mean().item():.6f}")

        intersection_points = ray_origins + dist.unsqueeze(-1) * ray_directions

        distances_xy = torch.norm(intersection_points[:, :2], dim=-1)
        within_cut = distances_xy <= self.cut_radius

        final_mask = ~active_mask & within_cut
        intersection_points[~final_mask] = float('nan')
        rays[~final_mask] = float('nan')

        return intersection_points.unsqueeze(1), rays, final_mask

# implemented nad checked but furthur check is required
class AsphereSurface(Surface):
    """
    """
    def __init__(self, center, radius, cut_radius, cone =0, coeffs = []):

        super().__init__()
        self.center = center 
        self.radius = radius 
        self.curvature = 1/self.radius 
        self.cut_radius = cut_radius  
        self.pos = center - torch.tensor([0,0,radius])
        self.conic = cone 
        self.coeffs = coeffs
        self.num_coeffs= coeffs.shape[0]

    def sag(self, points):
        r2 = points[...,0:1]**2 + points[...,1:2]**2
        surface = (
            r2 * (self.curvature) / (1 + torch.sqrt(1 - (1 + self.conic) * r2 * self.curvature**2 + 1e-8))
        )

        if self.num_coeffs > 0:
            for i in range(0, self.num_coeffs):
                surface = surface + self.coeffs[i]* 2 ** (i+1)

        return surface
    
    def normal(self, points):
        nx, ny, nz = self.first_deriv(points)
        n_vec = torch.cat((nx, ny, nz), axis=-1)
        n_vec = F.normalize(n_vec, dim=-1)
        norm = torch.cat([points, n_vec], axis=1)

        return norm
    
    def first_deriv(self, points):
        r2 = points[...,0:1]**2 + points[...,1:2]**2
        sf = torch.sqrt(1 - (1 + self.conic) * r2 * self.curvature**2 + 1e-8)
        dsdr2 = (
            (1 + sf + (1 + self.conic) * r2 * self.curvature**2 / 2 / sf) * self.curvature / (1 + sf) ** 2
        )
        if self.num_coeffs > 0:
            for i in range(0, self.num_coeffs):
                dsdr2 = dsdr2 + (i+1) * self.coeffs[i]* 2 ** i
        return dsdr2 * 2* points[...,0:1], dsdr2 * 2* points[...,1:2], - torch.ones_like(points[...,0:1])     
       

    def intersection(self, rays, number_of_steps=200, error_threshold=1e-5):

        ray_origins = rays[:, 0, :3] 
        ray_directions = F.normalize(rays[:, 1, :3], dim=-1)  # (N, 3)
        num_rays = ray_origins.shape[0]

        # Initial distance guesses
   
        dist = ((self.pos[...,2] - ray_origins[...,2]) / ray_directions[...,2]).squeeze(0)
        
        def compute_error_and_derivative(d):
            p = ray_origins + d.unsqueeze(-1) * ray_directions 
            error = self.sag(p[...,0:2]) + self.pos[...,2:3].squeeze(0) - p[...,2:3]

            deriv_x, deriv_y, deriv_z = self.first_deriv(p)

            d_error = deriv_x*ray_directions[...,0:1] + deriv_y*ray_directions[...,1:2] + deriv_z*ray_directions[...,2:3]

            return error.squeeze(), d_error.squeeze()

        t = tqdm(range(number_of_steps), leave=False, dynamic_ncols=True)

        active_mask = torch.ones(num_rays, dtype=torch.bool, device=ray_origins.device)

        for step in t:
            error, d_error = compute_error_and_derivative(dist)
            safe = d_error.abs() > 1e-8
            # d_next = d - f / f'
            delta = torch.where(safe, error / d_error, torch.zeros_like(d_error))
            dist = dist - delta
            dist = dist.clamp(min=0)
            active_mask &= error.squeeze().abs() > error_threshold
            if not active_mask.any():
                break

            t.set_description(f"Newton sphere intersection loss: {error[active_mask].mean().item():.6f}")

        intersection_points = ray_origins + dist.unsqueeze(-1) * ray_directions

        # Check for rays within XY cut radius
        distances_xy = torch.norm(intersection_points[:, :2], dim=-1)
        within_cut = distances_xy <= self.cut_radius

        final_mask = ~active_mask & within_cut
        intersection_points[~final_mask] = float('nan')
        rays[~final_mask] = float('nan')
        # print(intersection_points.shape)
        return intersection_points.unsqueeze(1), rays, final_mask
    
class SurfaceFactory:
    """
    Factory class to create surfaces based on type.
    """
    @staticmethod
    def create_surface(surface_data, pos, diameter):
        surface_type = surface_data["type"]
        if surface_type == "sphere":
            # print(surface_data)
            radius_of_curvature = surface_data["radius_of_curvature"]
            center = pos + torch.tensor([0, 0, radius_of_curvature])
            return SphereSurface(center.unsqueeze(0).unsqueeze(0), radius_of_curvature, diameter / 2)

        elif surface_type == "plano":
            return PlaneSurface(pos.unsqueeze(0).unsqueeze(0), torch.tensor([0.0, 0.0, -1.0]).unsqueeze(0).unsqueeze(0), diameter / 2)

        elif surface_type == "asphere":
            radius_of_curvature = surface_data["radius_of_curvature"]
            center = pos + torch.tensor([0, 0, radius_of_curvature])
            conic = torch.tensor(surface_data["k"])
            coeffs = torch.tensor(surface_data["coeffs"])
            return AsphereSurface(center.unsqueeze(0).unsqueeze(0), radius_of_curvature, diameter / 2,
                                 conic, coeffs)

        else:
            # free-form surfaces should be added later
            raise ValueError(f"Unknown surface type: {surface_type}")



class Lens:


    # MATERIAL_CAUCHY_COEFFICIENTS = {
    #     "air": [1.0003, 0.0, 0.0],
    #     "N-BK7": [1.5046, 0.0042, 0.000005],
    #     "BK7": [1.5046, 0.0042, 0.000005],
    #     "Fused Silica": [1.4580, 0.00354, 0.000003]
    # }

    MATERIAL_SELLMEIER_COEFFICIENTS = {
        "air": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "N-BK7": [1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653],
        "BK7": [1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653],
        "Fused Silica": [0.6961663, 0.4079426, 0.8974794, 0.0684043, 0.1162414, 9.896161]

    }
    MATERIAL_MAP = {
        "air": 1.0,
        "N-BK7": 1.5666,
        "BK7": 1.5168,
        "Fused Silica": 1.458,
        # Add more materials here as needed
    }

    def __init__(self, lens_dict,material_before, material_after, wavelength_nm=550):
        """
        Initialize the lens object and load the lens configuration.

        Args:
            lens_dict (dict): A dictionary containing the lens configuration.
        """
        self.wv = wavelength_nm
        self.load_lens(lens_dict)
        self.n0 = material_before  # Refractive index of the medium before the lens
        self.n2 = material_after  # Refractive index of the medium after the lens
        


    def sellmeier_refractive_index(self, material):

        if material not in self.MATERIAL_SELLMEIER_COEFFICIENTS:
            raise ValueError(f"Material '{material}' not found in database.")
        
        B1, B2, B3, C1, C2, C3 = self.MATERIAL_SELLMEIER_COEFFICIENTS[material]
        wavelength_um = self.wv / 1000  
        
        n_squared = 1 + (B1 * wavelength_um**2) / (wavelength_um**2 - C1) + \
                        (B2 * wavelength_um**2) / (wavelength_um**2 - C2) + \
                        (B3 * wavelength_um**2) / (wavelength_um**2 - C3)
        
        return n_squared**0.5

    def load_lens(self, lens_dict):
 
        material_name = lens_dict["material"] 
        # self.material = self.MATERIAL_MAP.get(material_name, None)
        self.material = self.sellmeier_refractive_index(material_name)
        # print('================================================================================')
        # print(self.material)
        # print('================================================================================')
        if self.material is None:
            raise ValueError(f"Material '{material_name}' not found in material map.")

        # Load front and back surfaces
        self.front_surface = SurfaceFactory.create_surface(
            lens_dict["surface1"],
            torch.tensor(lens_dict["xy-center"], dtype=torch.float64),
            lens_dict["diameter"]
        )

        self.back_surface = SurfaceFactory.create_surface(
            lens_dict["surface2"],
            torch.tensor(lens_dict["xy-center"], dtype=torch.float64) + torch.tensor([0.0, 0.0, lens_dict["tc"]], dtype=torch.float64),
            lens_dict["diameter"]
        )
        # print(self.back_surface)
        # exit()
        self.diameter = lens_dict["diameter"]
        self.pos = torch.tensor(lens_dict["xy-center"], dtype=torch.float64)
        self.th_edge = torch.tensor(lens_dict["te"], dtype=torch.float64)
        self.th_curve = torch.tensor( lens_dict["tc"], dtype=torch.float64)

    def apply(self, rays):
        """
        Returns:
            torch.Tensor: The transformed rays after passing through the lens. (num_rays, 2, 3)
        """
        intersection_front, intersected_rays_front, mask_front = self.front_surface.intersection(rays)

        valid_mask_front = ~torch.any(intersection_front.isnan(), dim=(1, 2))
        intersection_front = intersection_front[valid_mask_front]
        intersected_rays_front = intersected_rays_front[valid_mask_front]
        # if self.pos[1] ==0:
        #     print(intersected_rays_front.shape)
        #     exit()
        # print(valid_mask_front)
        refracted_front = self.front_surface.refract(intersected_rays_front, intersection_front, self.n0, self.material)
        intersection_back, intersected_rays_back, mask_back = self.back_surface.intersection(refracted_front)
        # print(intersected_rays_back.shape)
        valid_mask_back = ~torch.any(intersection_back.isnan(), dim=(1, 2))
        intersection_back = intersection_back[valid_mask_back]
        intersected_rays_back = intersected_rays_back[valid_mask_back]
        # print(intersected_rays_back.shape, intersection_back)

        refracted_back = self.back_surface.refract(intersected_rays_back, intersection_back, self.material, self.n2)
        # print(refracted_back.shape)

        return refracted_back



    #no asphere yet
    def sample(self, num_samples=50000):
            
        if (type(self.front_surface) == PlaneSurface):
            # Sample uniform radius with area weighting
            r = self.front_surface.radius * torch.sqrt(torch.rand(num_samples, dtype=torch.float64))

            theta = torch.rand(num_samples, dtype=torch.float64) * (2 * torch.pi )

            # Convert to Cartesian coordinates
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full((num_samples, 1), self.pos[2].item(), dtype=torch.float64)

        elif type(self.front_surface) == SphereSurface:
            theta = torch.rand(num_samples, dtype=torch.float64) * (2 * torch.pi)
            phi_min = torch.asin(torch.tensor((self.front_surface.cut_radius)/self.front_surface.radius))
            cos_phi_min = torch.cos(torch.tensor(phi_min, dtype=torch.float64))
            cos_phi_max = torch.cos(torch.tensor(torch.pi*2, dtype=torch.float64))
            cos_phi = torch.rand(num_samples, dtype=torch.float64) * (cos_phi_min - cos_phi_max) + cos_phi_max
            phi = torch.acos(cos_phi)
            x = self.front_surface.radius * torch.cos(theta) * torch.sin(phi)
            y = self.front_surface.radius * torch.sin(theta) * torch.sin(phi)
            # print(x.min(), x.max())
            # print(y.min(), y.max())
            # z = self.front_surface.radius * torch.cos(phi) - (self.front_surface.radius - (self.th_curve - self.th_edge)/2 if type(self.back_surface) == SphereSurface else (self.th_curve - self.th_edge))
            z = -1 *(self.front_surface.radius * torch.cos(phi) - self.front_surface.radius) + \
                                                            self.front_surface.pos[0,0,2]

        else:
            raise ValueError("Invalid surface type. Choose 'flat' or 'sphere'.")

        points_in_circle = torch.stack((x, y, z.squeeze()), dim=1).unsqueeze(1)
        #fix the issue 
        points_in_circle[...,:2] = points_in_circle[...,:2] + self.front_surface.pos[...,:2]


        return points_in_circle

    def __call__(self, rays):
        return self.apply(rays)

class LensList:
    """
    A container class to hold a list of Lens objects and apply their transformations sequentially.
    """

    def __init__(self):
        self.lenses = []

    def add_lens(self, lens):

        if isinstance(lens, Lens):
            self.lenses.append(lens)
        else:
            raise TypeError("Only objects of type 'Lens' can be added to LensList.")

    def __getitem__(self, index):

        if index < 0:
            index += len(self.lenses)

        if 0 <= index < len(self.lenses):
            return self.lenses[index]
        else:
            raise IndexError("Lens index out of range.")


class ProjectionSys:

    def __init__(self, sensor_dict, lens_dicts, aperture_dict, mode = 'row', object_dict=None, fl = 0, wavelength_nm=550):
        """
           Arg mode decides whether we have lens array  or a series of lens one after the other
        """
        self.sensor = Sensor(sensor_dict)
        self.wv = wavelength_nm
        self.lens_list = LensList()
        self.mode = mode
        self.fl = fl
        for index, lens_dict in enumerate(lens_dicts):
            material_before = Lens.MATERIAL_MAP['air']
            material_after = Lens.MATERIAL_MAP['air']

            if index > 0:
                previous_lens = lens_dicts[index - 1]
                close = previous_lens['xy-center'][2] + previous_lens['tc']
                if lens_dict['xy-center'] == close:
                    material_before = Lens.MATERIAL_MAP[previous_lens['material']]

            if index < (len(lens_dicts) - 1):
                next_lens = lens_dicts[index + 1]
                later = next_lens['xy-center']
                if (lens_dict['xy-center'][2] + lens_dict['tc']) == later:
                    material_after = Lens.MATERIAL_MAP[next_lens['material']]

            self.lens_list.add_lens(Lens(lens_dict, material_before, material_after, wavelength_nm = self.wv))


        #for now object is just a circular plane
        self.image_distance = self.lens_list[0].front_surface.pos[0,0,2]

        self.target = SurfaceFactory.create_surface(object_dict,
                                                    torch.tensor(self.lens_list[-1].back_surface.pos + torch.tensor([0.0,0.0, object_dict['distance_before'] if self.fl == 0 else 1/(1/self.fl - 1/self.image_distance)] ),
                                                    dtype=torch.float64).squeeze().squeeze(), torch.tensor(object_dict['diameter'], dtype=torch.float64))


        # self.pupil = torch.tensor(aperture_dict['xy_center'], dtype=torch.float64)  # Initialize pupil with [0, 0, 0]
        # self.aperture_size = PlaneSurfacetorch.tensor(aperture_dict['diameter'], dtype=torch.float64)
        self.pupil = SurfaceFactory.create_surface({'type': 'plano'},  torch.tensor(aperture_dict['xy-center'], dtype=torch.float64), torch.tensor(aperture_dict['diameter'], dtype=torch.float64))
        self.object_distance = self.target.pos[0,0,2] - self.lens_list[-1].back_surface.pos[0,0,2]


    def create_object(self, object_dict):
        return object_dict

    # try to implement chief ray but its not correct, furthur study is needed
    # def _check_and_exitpupil(self, rays, ref_origin, ref_dir, num_pixels):
    #     """
    #     Combines intersection check and exit pupil determination for rays.

    #     Args:
    #     - rays (torch.Tensor): A tensor of rays, shape (num_rays, 2, 7).
    #     - ref_origin (torch.Tensor): Origin of the reference ray (shape: (3,)).
    #     - ref_dir (torch.Tensor): Direction of the reference ray (shape: (3,)).
    #     - num_pixels (int): Number of pixels in the sensor.

    #     Returns:
    #     - rays (torch.Tensor): Updated rays with flags set for valid intersections.
    #     - intersection_points (torch.Tensor): Intersection points for rays; [0, 0, inf] for non-intersecting rays.
    #     - exit_pupil_found (bool): Whether an exit pupil has been found.
    #     """
    #     if not torch.allclose(self.pupil, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)):
    #         return rays, self.pupil.unsqueeze(0).repeat(rays.size(0), 1), True

    #     origins = rays[:, 0, :3]  # (num_rays, 3)
    #     directions = rays[:, 1, :3]  # (num_rays, 3)
    #     lx, ly, lz = origins[:, 0], origins[:, 1], origins[:, 2]
    #     dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    #     t_x = lx / dx
    #     t_y = ly / dy

    #     # Identify rays that are nearly parallel (small difference between t_x and t_y)
    #     t_values = torch.where(torch.abs(t_x - t_y) < 1e-5, t_x, torch.tensor(float('nan')))

    #     # Handle parallel rays
    #     parallel_mask = (dx.abs() < 1e-6) & (dy.abs() < 1e-6)

    #     # If the ray is parallel but its origin matches the ref_origin, calculate (0, 0, s_values)
    #     origin_matches_ref = torch.all(torch.isclose(origins, ref_origin), dim=1)
    #     intersection_points = torch.stack([lx + t_values * dx, ly + t_values * dy, lz + t_values * dz], dim=-1)
    #     # Apply the specific handling for parallel rays
    #     parall_mask = parallel_mask & origin_matches_ref
    #     intersection_points[parallel_mask & ~origin_matches_ref] = torch.tensor([0.0, 0.0, float('inf')], dtype=torch.float64)

    #     intersection_points[parallel_mask & origin_matches_ref] = torch.stack([
    #             torch.full((intersection_points[parallel_mask & origin_matches_ref].shape[0],), ref_origin[0], dtype=torch.float64),
    #             torch.full((intersection_points[parallel_mask & origin_matches_ref].shape[0],), ref_origin[1], dtype=torch.float64),
    #             (lz + t_values * dz)[parallel_mask & origin_matches_ref]
    #         ], dim=-1)

    #     # Handle invalid intersections (when dz is zero for non-parallel rays)
    #     intersection_mask = torch.isfinite(lz + t_values * dz)
    #     valid_points = intersection_points[intersection_mask]
    #     valid_rays = rays[intersection_mask]
    #     if valid_points.size(0) == 0:
    #         intersection_points = torch.full_like(intersection_points, float('inf'))
    #         return rays, intersection_points, False

    #     # Find unique intersection points
    #     unique_points, inverse_indices = torch.unique(valid_points, dim=0, return_inverse=True)
    #     group_counts = torch.bincount(inverse_indices)

    #     # Check if we have a consistent intersection for all rays
    #     matching_group_idx = (group_counts == num_pixels).nonzero(as_tuple=True)[0]
    #     if matching_group_idx.numel() > 0:
    #         matching_point = unique_points[matching_group_idx[0]]
    #         matching_mask = inverse_indices == matching_group_idx[0]

    #         rays[intersection_mask, 0, 6] = matching_mask.to(torch.float64)
    #         rays[intersection_mask, 1, 6] = matching_mask.to(torch.float64)
    #         self.pupil = matching_point
    #         exit_pupil_found = True
    #     else:
    #         matching_point = torch.tensor([0.0, 0.0, float('inf')], dtype=torch.float64)
    #         exit_pupil_found = False

    #     # Replace intersection points with [0, 0, inf] for non-intersecting rays
    #     intersection_points = torch.where(
    #         intersection_mask.unsqueeze(1),
    #         intersection_points,
    #         torch.tensor([0.0, 0.0, float('inf')], dtype=torch.float64)
    #     )
    #     return rays, intersection_points, exit_pupil_found

    def check_aperture(self, rays):


        intersection_points, intersected_rays, mask = self.pupil.intersection(rays)

        pupil_pos = self.pupil.center.view(1, 1, 3)

        distances = torch.norm(intersection_points[:,0,:2]- pupil_pos[:,0, :2], dim = -1)
        unique_parents, inverse_indices = torch.unique(intersected_rays[:, 0, 3:6], dim=0, return_inverse=True)

        unique_categories = torch.arange(unique_parents.shape[0], dtype= torch.float64)

        masks = (inverse_indices == unique_categories.view(-1, 1))  

        separated_values = distances * (masks.float() * 2 - 1)  
        separated_values[separated_values <0] = float('inf')
        min_values, min_indices = torch.min(separated_values, dim=-1)

        closest = torch.zeros_like(distances)
        closest[min_indices] = 1.0
        closeness = torch.all(torch.isclose(intersection_points, pupil_pos, atol=1), dim=-1).squeeze()
        matching_mask = closeness.bool() & closest.bool()
        intersected_rays[:, 0, 6] = matching_mask.squeeze(-1).to(torch.float64)
        intersected_rays[:, 1, 6] = matching_mask.squeeze(-1).to(torch.float64)

        return intersected_rays, True

    def aperture_sample(self, num_samples=50000):
        r = self.pupil.radius * torch.sqrt(torch.rand(num_samples, dtype=torch.float64))

        theta = torch.rand(num_samples, dtype=torch.float64) * (2 * torch.pi )

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.full((num_samples, 1), self.pupil.pos[0,0,2].item(), dtype=torch.float64)
        samples = torch.stack((x, y, z.squeeze()), dim=1).unsqueeze(1)
        samples[...,:2] = samples[...,:2]+ self.pupil.pos[...,:2]
        return samples 


    def forward(self):
        """
        Traces rays through the projection system from the first lens to the sensor.

        Steps:
        1. Samples rays using the first lens in the lens list.
        2. Processes those rays using the sensor.
        3. Iteratively passes the rays through the lens list, and checks for exit pupil.

        Returns:
        - Final rays after processing through the system.
        """
        if not self.lens_list.lenses:
            raise ValueError("Lens list is empty; cannot trace rays.")

        # whether you want to have points on surface of lens or on aperture
        # points = self.lens_list.lenses[0].sample(num_samples = self.sensor.raysperpix ) if self.mode =='row' else self.aperture_sample(num_samples = self.sensor.raysperpix)
        points = self.aperture_sample(num_samples = self.sensor.raysperpix)
        rays, origin = self.sensor(points)

        aperture_check = False
        rays_list = []
        for i, lens in enumerate(self.lens_list.lenses):
            print('---------------------------{}----------------------'.format(i))
            if (not aperture_check) and (self.pupil.center.squeeze(1).squeeze(0)[-1] < lens.pos[-1]):
                rays, aperture_check = self.check_aperture(rays)
                print(rays.shape)
            print('----------newlens_______________')
            print(rays.shape)
            print(lens.material)
            out_rays = lens(rays)
            print('-----------afterlens------------')
            print(out_rays.shape)
            print('----------forexitcheck__________')

            if self.mode == 'row':
                rays = out_rays
            else:
                print(out_rays.shape)
                rays_list.append(out_rays)
                # exit()
        if self.mode == 'array':
            rays = torch.cat(rays_list, dim=0)
        print('---------------------------------------------------------------------------------')
        if not aperture_check:
            rays, _ = self.check_aperture(rays)
        f_intersection, rays, _ = self.target.intersection(rays)
        # some stupid conding here, I changed the ray origin here for the last rays as it did not follow my previous code logic
        rays[:, 0:1, :3] = f_intersection
        print(rays.shape)
        wvs = torch.full_like(rays[0:1, :, :1], self.wv)
        rays = torch.cat((rays, wvs.expand(rays.size(0), -1, -1)), dim=2)

        return f_intersection,  rays, self.image_distance, self.object_distance


#Relationship between the image size and the pixel size should be reconsidered, currently it is problematic when it comes to centered one
# produced but not used
class Render:
    def __init__(self, image_size_mm, image_size_px = [1001,1001], kernel_size = [1001, 1001], centered = False, positional = False, output_directory="test_output"):
        self.dpm = image_size_px[0]/ image_size_mm[0] if (not centered) else kernel_size[0]/ image_size_mm[0]

        self.image_size_mm = image_size_mm
        self.image_size_px = image_size_px
        self.output_directory = output_directory
        self.centered = centered
        self.positional = positional
        self.kernel_size = kernel_size
        os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

        self.image_center_mm = torch.tensor(
            [image_size_mm[0] / 2, image_size_mm[1] / 2], dtype=torch.float64)

    def mm_to_pixel(self, points_mm):
        # print(points_mm)

        image_center_mm = torch.tensor([self.image_size_mm[0] / 2, self.image_size_mm[1] / 2], dtype=torch.float64)
        image_center_px = torch.tensor([-1*self.image_size_mm[0]/ self.image_size_px[0]/ 2, 1* self.image_size_mm[1]/self.image_size_px[1] / 2], dtype=torch.float64)
        points_new= (points_mm[:, 0, :2] + image_center_mm)
        points_new[:, 0] = self.image_size_mm[0] - points_new[:, 0]
        points_pixel =  points_new * self.dpm * 0.9999
        print(points_pixel)
        print(points_new * self.dpm )

        return points_pixel

    @staticmethod
    def bilinear_interpolation(image, points):
        h, w = image.shape
        x, y = points[:, 0], points[:, 1]

        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0.float(), y - y0.float()

        w00, w01 = (1 - dx) * (1 - dy), dx * (1 - dy)
        w10, w11 = (1 - dx) * dy, dx * dy


        indices = torch.stack([y0, x0], dim=1).t()

        image.index_put_((indices[0], indices[1]), w00, accumulate=True)

        indices = torch.stack([y0, x1], dim=1).t()
        image.index_put_((indices[0], indices[1]), w01, accumulate=True)

        indices = torch.stack([y1, x0], dim=1).t()
        image.index_put_((indices[0], indices[1]), w10, accumulate=True)

        indices = torch.stack([y1, x1], dim=1).t()
        image.index_put_((indices[0], indices[1]), w11, accumulate=True)

        return image

    def create_image(self, image_size_px):
        return torch.zeros((image_size_px[1], image_size_px[0]), dtype=torch.float64)

    def render(self, points_mm, image_size_px, file_name=None):
        points_px = self.mm_to_pixel(points_mm[~torch.isnan(points_mm).all(dim=(1, 2))])
        image = self.create_image(image_size_px)
        image = self.bilinear_interpolation(image, points_px)
        print(torch.sum(image[image>0]))
        if not file_name:
            file_name = 'spsf.png'
        return [self.RenderedImage(image, self.output_directory, file_name)], None

    def centered_render(self, points_mm, rays, image_distance, object_distance):

        no_null_points = points_mm[~torch.isnan(points_mm).all(dim=(1, 2))]
        no_null_rays = rays[~torch.isnan(points_mm).all(dim=(1, 2))]

        parent_positions = no_null_rays[:, 0, 3:6]

        unique_parents, inverse_indices = torch.unique(parent_positions, dim=0, return_inverse=True)

        centered_psfs = []
        psfs_image = []
        for i in range(unique_parents.shape[0]):

            # Find the rays in the current group
            group_mask = inverse_indices == i
            group_rays = no_null_rays[group_mask]
            group_points = no_null_points[group_mask]
            print('--------------------------------------------------------------------------------')
            # print(group_points)
            idx_of_one = group_rays[:, 0, 6] == 1
            # print(idx_of_one)

            if idx_of_one.any():
                reference_point = group_points[idx_of_one]
                # print(reference_point)
                shifted_group_points = group_points  -reference_point
                centered, _ = self.render(shifted_group_points, self.kernel_size, \
                                                             file_name = "centered_psf_{}_{}_{}_{}.png".format(image_distance, object_distance, unique_parents[i,0], unique_parents[i,1]))
                centered_psfs.append(centered[0])
                psfs_image.append(centered[0].image_tensor)

        return centered_psfs, torch.stack(psfs_image)

    def positional_render(self, points_mm, rays, image_distance, object_distance):

        no_null_points = points_mm[~torch.isnan(points_mm).all(dim=(1, 2))]
        no_null_rays = rays[~torch.isnan(points_mm).all(dim=(1, 2))]

        parent_positions = no_null_rays[:, 0, 3:6]

        unique_parents, inverse_indices = torch.unique(parent_positions, dim=0, return_inverse=True)

        psfs = []
        psfs_image = []
        for i in range(unique_parents.shape[0]):

            # Find the points in the current group
            group_mask = inverse_indices == i
            group_points = no_null_points[group_mask]

            print('--------------------------------------------------------------------------------')
            # print(group_points)

            psf, _ = self.render(group_points, self.image_size_px,
                            file_name="positional_psf_{}_{}_{}_{}.png".format(image_distance, object_distance, unique_parents[i, 0], unique_parents[i, 1]))
            psfs.append(psf[0])
            psfs_image.append(psf[0].image_tensor)
        
        return psfs, torch.stack(psfs_image)

    def __call__(self, points_mm, rays=None, image_distance=0, object_distance=0):
        if not self.centered and not self.positional:
            return self.render(points_mm, self.image_size_px)
        elif self.centered:
            return self.centered_render(points_mm, rays, image_distance, object_distance)
        elif self.positional:
            return self.positional_render(points_mm, rays, image_distance, object_distance)

    class RenderedImage:
        def __init__(self, image_tensor, output_directory, file_name):
            self.image_tensor = image_tensor
            self.output_directory = output_directory
            self.file_name = file_name
        def save_image(self):

            # self.image_tensor = (self.image_tensor - self.image_tensor.mean()) / self.image_tensor.std()
            psf_gray_scale = (self.image_tensor - self.image_tensor.min()) / \
                              (self.image_tensor.max() - self.image_tensor.min()) * 255
            psf_gray_scale = psf_gray_scale.cpu().numpy().astype(np.uint8)
            image = Image.fromarray(psf_gray_scale, mode = 'L')
            image.save(os.path.join(self.output_directory, self.file_name))
