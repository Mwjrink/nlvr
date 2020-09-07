// use winit::monitor::VideoMode;
// use winit::window::Fullscreen::Exclusive;
#[macro_use]
extern crate log;
use simple_logger::SimpleLogger;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::time::{Duration, Instant};

use nlvr::lib::vulkan;

fn main() {
    /* simple logger */
    SimpleLogger::new().init().unwrap();
    /* simple logger */

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_resizable(false)
        .with_title("Vulkan, Hello World")
        // .with_fullscreen(Some(Exclusive(VideoMode { })))
        // .with_decorations(false)
        .build(&event_loop)
        .unwrap();

    let app = vulkan::VulkanApp::new(&window);

    // event_loop.available_monitors();

    let mut avg = Duration::from_millis(0);
    let mut last = Instant::now();

    let mut dirty_swapchain = true;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                // frame timing info
                let now = Instant::now();
                let delta = now.duration_since(last);
                last = now;
                avg = (avg + delta) / 2;
                // window.push_back(delta);
                // println!("{:?}", avg.as_nanos());
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
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
            }
            _ => (),
        }
    });
}
