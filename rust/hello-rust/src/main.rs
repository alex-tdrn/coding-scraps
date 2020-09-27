//extern crate glfw;

use glfw::{Action, Context, Key, WindowEvent};

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (mut window, events) = glfw
        .create_window(300, 300, "Hello this is window", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.make_current();
    window.set_key_polling(true);
    window.set_resizable(true);

    while !window.should_close() {
        window.swap_buffers(); 
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            println!("{:?}", event);
            match event {
                WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
                WindowEvent::FramebufferSize(x, y) => window.set_size(x, y),
                _ => {}
            }
        }
    }
}
