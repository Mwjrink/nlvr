use cgmath::{vec3, Matrix4};
use nlvr::lib::{camera::Camera, math, renderinstance::*, ubo::CameraUBO};

// use winit::monitor::VideoMode;
// use winit::window::Fullscreen::Exclusive;
extern crate log;
extern crate vk_mem;

use simple_logger::SimpleLogger;

use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use cgmath::{Deg, Point3, Vector3};

use std::time::Instant;

const FRAMES_AVERAGE: u32 = 5;

// use vulkan;

// #[macro_use] extern crate shrinkwraprs;
//
// #[derive(Shrinkwrap)]
// struct Email(String);

/*
TODO
 - Use shrinkwraprs


*/

fn main() {
    // simple logger
    SimpleLogger::new().init().unwrap();
    // simple logger

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_resizable(true)
        .with_title("Vulkan, Hello World")
        // .with_fullscreen(Some(Exclusive(VideoMode { })))
        // .with_decorations(false)
        .build(&event_loop)
        .unwrap();

    // let mut app = VulkanApp::new(&window,);

    let mut render_instance = RenderInstance::<CameraUBO>::create(&window);
    let cottage_renderable =
        render_instance.renderable_from_file("chalet/chalet.obj".to_string(), "chalet/chalet.jpg".to_string());

    // let rock_assembly_cliffs_renderable = render_instance.renderable_from_file(
    // "quixel/Rock_Assembly_Cliffs_siEoZ_8K_3d_ms/siEoZ_High.obj".to_string(),
    // None,
    // );
    let fire_pit_renderable = render_instance.renderable_from_file(
        "quixel/fire_pit/fire_pit.obj".to_string(),
        "quixel/fire_pit/fire_pit_albedo.jpg".to_string(),
    );

    let base_rot = Matrix4::from_angle_x(Deg(270.0));
    let transform_0 = Matrix4::from_translation(vec3(0.1, 0.0, -1.0)) * base_rot;
    let transform_1 = Matrix4::from_translation(vec3(-0.1, 0.0, 1.0)) * Matrix4::from_scale(0.01);
    // let transform_2 = Matrix4::from_translation(vec3(0.0, 0.0, 0.0)) * base_rot * Matrix4::from_scale(0.01);

    let _cottage_renderable_instance_0 = render_instance
        .get_renderable(cottage_renderable)
        .create_instance(transform_0);
    // let cottage_renderable_instance_1 = render_instance
    //     .get_renderable(cottage_renderable)
    //     .create_instance(transform_1);

    let _fire_pit_renderable_instance_0 = render_instance
        .get_renderable(fire_pit_renderable)
        .create_instance(transform_1);

    // let rock_assembly_cliffs_renderable_instance_0 = render_instance
    // .get_renderable(rock_assembly_cliffs_renderable)
    // .create_instance(transform_2);

    let mut camera = Camera::default();

    // event_loop.available_monitors();

    let mut frames = vec![0.0; FRAMES_AVERAGE as usize];
    let mut frame_index = 0;
    let mut last = Instant::now();
    let mut fps: f64 = 0.0;

    let mut dirty_swapchain = true;

    // Used to accumutate input events from the start to the end of a frame
    let mut is_left_clicked = false;
    let mut cursor_position: [i32; 2] = [0, 0];
    let mut last_position: [i32; 2] = cursor_position;
    let mut wheel_delta: f32 = 0.0;

    let mut dt = 0.0;
    let mut seconds = 5.0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                // reset input states on new frame
                {
                    // is_left_clicked = false;
                    last_position = cursor_position;
                    // cursor_position = [0, 0];
                    wheel_delta = 0.0;
                }
                // frame timing info
                let now = Instant::now();
                let delta = now.duration_since(last);
                last = now;
                dt = delta.as_secs_f32();
                frames[frame_index] = delta.as_secs_f64();
                frame_index = (frame_index + 1) % (FRAMES_AVERAGE as usize);
                fps = f64::from(FRAMES_AVERAGE) / frames.iter().sum::<f64>();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                // update input state after accumulating event
                // {
                //     if let Some(is_left_clicked) = is_left_clicked {
                //         is_left_clicked = is_left_clicked;
                //     }
                //     if let Some(position) = cursor_position {
                //         cursor_position = position;
                //         cursor_delta = Some([position[0] - last_position[0], position[1] - last_position[1]]);
                //     } else {
                //         cursor_delta = None;
                //     }
                //     wheel_delta = wheel_delta;
                // }

                // render
                {
                    if dirty_swapchain {
                        let size = window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            // app.recreate_swapchain();
                            render_instance.rebuild();
                        } else {
                            return;
                        }
                    }

                    let width = render_instance.render_width();
                    let height = render_instance.render_height();

                    let delta = [
                        cursor_position[0] - last_position[0],
                        cursor_position[1] - last_position[1],
                    ];

                    // translate

                    // Update uniform buffers
                    let ubo = {
                        if is_left_clicked && (delta[0] != 0 && delta[1] != 0) {
                            let x_ratio = delta[0] as f32 / width as f32;
                            let y_ratio = delta[1] as f32 / height as f32;
                            let theta = x_ratio * 180.0_f32.to_radians();
                            let phi = y_ratio * 90.0_f32.to_radians();
                            camera.rotate(theta, phi);
                        }

                        if wheel_delta != 0.0 {
                            camera.forward(wheel_delta * 0.3);
                        }

                        let aspect = width as f32 / height as f32;
                        let ubo = CameraUBO {
                            view: Matrix4::look_at_rh(
                                camera.position(),
                                Point3::new(0.0, 0.0, 0.0),
                                Vector3::new(0.0, 1.0, 0.0),
                            ),
                            proj: math::perspective(Deg(45.0), aspect, 0.1, 10.0),
                        };

                        ubo
                    };

                    render_instance.update_uniform_buffers(ubo);
                    dirty_swapchain = render_instance.draw_frame();
                }

                seconds -= dt;
                if seconds <= 0.0 {
                    println!(
                        "FPS: {} \nCursor_Position: [{}, {}] \nLeft_Clicked: {}",
                        fps, cursor_position[0], cursor_position[1], is_left_clicked
                    );

                    seconds = 5.0;
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized { .. } => dirty_swapchain = true,
                // Accumulate input events
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    if state == ElementState::Released {
                        is_left_clicked = false;
                    }
                    if state == ElementState::Pressed {
                        is_left_clicked = true;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let position: (i32, i32) = position.into();
                    cursor_position = [position.0, position.1];
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, v_lines),
                    ..
                } => {
                    wheel_delta = v_lines;
                }
                _ => (),
            },
            Event::LoopDestroyed => {
                render_instance.wait_gpu_idle();
                println!("completed");
            }
            _ => (),
        }
    });
}

// TODO:
// - start splitting up (into files) and abstracting the vulkan operations (files like, pipeline, command buffer, queue
//   (graphics, present and compute), swapchain, image, image_view, descriptor set, command pool, texture, vector,
//   index, uniform something, descriptor pool etc)
// - model and textures along with their location, rotation scale provided from main
// - input and camera are bad, fix them
// - use a computation pipeline before the render pipeline and use the generated nlvo* files
// - swapvec ecs system
