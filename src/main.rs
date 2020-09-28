use nlvr::lib::vulkan::*;

// use winit::monitor::VideoMode;
// use winit::window::Fullscreen::Exclusive;
extern crate log;
use simple_logger::SimpleLogger;

use winit::{event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
            event_loop::{ControlFlow, EventLoop},
            window::WindowBuilder};

use std::time::{Duration, Instant};

// use vulkan;

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

    let mut app = VulkanApp::new(&window,);

    // event_loop.available_monitors();

    let mut avg = Duration::from_millis(0,);
    let mut last = Instant::now();

    let mut dirty_swapchain = true;

    // Used to accumutate input events from the start to the end of a frame
    let mut is_left_clicked = None;
    let mut cursor_position = None;
    let mut last_position = app.cursor_position;
    let mut wheel_delta = None;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::NewEvents(_,) => {
                // reset input states on new frame
                {
                    is_left_clicked = None;
                    cursor_position = None;
                    last_position = app.cursor_position;
                    wheel_delta = None;
                }
                // frame timing info
                let now = Instant::now();
                let delta = now.duration_since(last,);
                last = now;
                avg = (avg + delta) / 2;
                // window.push_back(delta);
                // println!("{:?}", avg.as_nanos());
            },
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                // update input state after accumulating event
                {
                    if let Some(is_left_clicked,) = is_left_clicked {
                        app.is_left_clicked = is_left_clicked;
                    }
                    if let Some(position,) = cursor_position {
                        app.cursor_position = position;
                        app.cursor_delta = Some([position[0] - last_position[0], position[1] - last_position[1],],);
                    } else {
                        app.cursor_delta = None;
                    }
                    app.wheel_delta = wheel_delta;
                }

                // render
                {
                    if dirty_swapchain {
                        let size = window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            app.recreate_swapchain();
                        } else {
                            return;
                        }
                    }
                    dirty_swapchain = app.draw_frame();
                }
            },
            Event::WindowEvent {
                event, ..
            } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized {
                    ..
                } => dirty_swapchain = true,
                // Accumulate input events
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    if state == ElementState::Pressed {
                        is_left_clicked = Some(true,);
                    } else {
                        is_left_clicked = Some(false,);
                    }
                },
                WindowEvent::CursorMoved {
                    position, ..
                } => {
                    let position: (i32, i32,) = position.into();
                    cursor_position = Some([position.0, position.1,],);
                },
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, v_lines,),
                    ..
                } => {
                    wheel_delta = Some(v_lines,);
                },
                _ => (),
            },
            Event::LoopDestroyed => app.wait_gpu_idle(),
            _ => (),
        }
    },);
}

// TODO:
// - start splitting up (into files) and abstracting the vulkan operations (files like, pipeline, command buffer, queue
//   (graphics, present and compute), swapchain, image, image_view, descriptor set, command pool, texture, vector,
//   index, uniform something, descriptor pool etc)
// - model and textures along with their location, rotation scale provided from main
// - input and camera are bad, fix them
// - use a computation pipeline before the render pipeline and use the generated nlvo* files
// - swapvec ecs system
