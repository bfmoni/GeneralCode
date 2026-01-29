import { useEffect, useRef } from "react"
import { useMap } from "./map"
import type BaseEvent from "ol/events/Event"
import Map from "ol/Map"

const vertexShaderSource = `#version 300 es
in vec2 vertex;
in vec2 offSet;

//bottom left corner, width span, height span
uniform vec4 map_bounds;
uniform float radius;

// all shaders have a main function
void main() {
  vec2 pos = vec2( (offSet[0] * radius + vertex[0]) , (offSet[1] * radius + vertex[1]));
  gl_Position = vec4(((pos[0] - map_bounds[0]) / map_bounds[2]) - 1.0, ((pos[1] - map_bounds[1]) / map_bounds[3]) -1.0 ,0.0,1.0);
}
`

const fragmentShaderSource = `#version 300 es
precision mediump float;
out vec4 outColor;
void main() {
  outColor = vec4(1, 1, 0.5, 1);
}
`

function makeCirclePattern() {
    const f_x = Math.cos(1 / 8 * Math.PI)
    const m_x = Math.cos(2 / 8 * Math.PI)
    const c_x = Math.cos(3 / 8 * Math.PI)
    const f_y = Math.sin(1 / 8 * Math.PI)
    const m_y = Math.sin(2 / 8 * Math.PI)
    const c_y = Math.sin(3 / 8 * Math.PI)

    return new Float32Array([
        0, 0,
        1, 0,
        f_x, f_y,
        m_x, m_y,
        c_x, c_y,
        0, 1,
        -c_x, c_y,
        -m_x, m_y,
        -f_x, f_y,
        -1, 0,
        -f_x, -f_y,
        -m_x, -m_y,
        -c_x, -c_y,
        0, -1,
        c_x, -c_y,
        m_x, -m_y,
        f_x, -f_y,
    ])
}

const circlePattern = makeCirclePattern()

const circleOrder = [
    0, 1, 2, //1
    0, 2, 3, //2
    0, 3, 4, //3
    0, 4, 5, //4
    0, 5, 6, //5
    0, 6, 7, //6
    0, 7, 8, //7
    0, 8, 9, //8
    0, 9, 10, //9
    0, 10, 11,//10
    0, 11, 12, //11
    0, 12, 13, //12
    0, 13, 14, //13
    0, 14, 15, //14
    0, 15, 16,//15
    0, 16, 1,//16
]

function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | false {
    const shader = gl.createShader(type)!
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS)
    if (success) {
        return shader
    }
    console.log(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader)
    return false
}

function createProgram(gl: WebGL2RenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram | false {
    var program = gl.createProgram()
    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)
    var success = gl.getProgramParameter(program, gl.LINK_STATUS)
    if (success) {
        return program
    }
    console.log(gl.getProgramInfoLog(program));
    gl.deleteShader(vertexShader)
    gl.deleteShader(fragmentShader)
    gl.deleteProgram(program);
    return false

}

function setupCircleVertexs(gl: WebGL2RenderingContext, program: WebGLProgram) {
    const positionLocation = gl.getAttribLocation(program, "offSet")
    //make a buffer for our positions
    const positionBuffer = gl.createBuffer()
    //bind it to array buffer position
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    //converts lat,lon array to gl buffer
    gl.bufferData(gl.ARRAY_BUFFER, circlePattern, gl.STATIC_DRAW)
    //enable our in data
    gl.enableVertexAttribArray(positionLocation)
    //THIS BINDS TO gl.ARRAY_BUFFER
    gl.vertexAttribPointer(
        positionLocation,
        2,
        gl.FLOAT,
        false,
        0,
        0
    )
}

function setupCirclePattern(gl: WebGL2RenderingContext) {
    const indexBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(circleOrder), gl.STATIC_DRAW)
}

function setupCircleCenter(gl: WebGL2RenderingContext, program: WebGLProgram, d: Float32Array) {
    const centerLocation = gl.getAttribLocation(program, "vertex")
    //make a buffer for our positions
    const offSetBuffer = gl.createBuffer()
    //bind it to array buffer position
    gl.bindBuffer(gl.ARRAY_BUFFER, offSetBuffer)
    //converts lat,lon array to gl buffer
    gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW)
    //enable our in data
    gl.enableVertexAttribArray(centerLocation)
    //THIS BINDS TO gl.ARRAY_BUFFER
    gl.vertexAttribPointer(
        centerLocation,
        2,
        gl.FLOAT,
        false,
        8,
        0
    )
    //only take one from here per instance
    gl.vertexAttribDivisor(centerLocation, 1)

    gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW)
}

function generateData(): Float32Array {
    const output = new Float32Array(200_000_000)
    for (let i = 0; i < output.length; i += 2) {
        output[i] = Math.random() * 180 * (Math.random() < .5 ? -1 : 1)
        output[i + 1] = Math.random() * 90 * (Math.random() < .5 ? -1 : 1)
    }
    return output
}

function makeProgram(gl: WebGL2RenderingContext): WebGLProgram | false {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    if (!vertexShader) {
        return false
    }
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)
    if (!fragmentShader) {
        return false
    }
    return createProgram(gl, vertexShader, fragmentShader)
}

function resize(gl: WebGL2RenderingContext, map: Map) {
    const bounds = map.getSize()!
    gl.canvas.width = bounds[0]
    gl.canvas.height = bounds[1]
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
}


function render(gl: WebGL2RenderingContext, program: WebGLProgram, numPoints: number, bl: [number, number], tr: [number, number], r: number) {
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    const boundsPosition = gl.getUniformLocation(program, "map_bounds")
    gl.uniform4f(boundsPosition, bl[0], bl[1], (tr[0] - bl[0]) / 2, (tr[1] - bl[1]) / 2)
    const radiusPosition = gl.getUniformLocation(program, "radius")
    gl.uniform1f(radiusPosition, r)
    gl.drawElementsInstanced(gl.TRIANGLES, 48, gl.UNSIGNED_SHORT, 0, numPoints)
}

export function WebGL() {
    const loaded = useRef(false)
    const data = useRef<Float32Array>(null)
    const gl = useRef<WebGL2RenderingContext | null>(null)
    const radius = useRef(0.005)
    const { map } = useMap()

    useEffect(() => {
        if (!loaded.current && map) {
            loaded.current = true
            const canvas = document.createElement("canvas")
            canvas.id = "glCanvas"
            canvas.style.zIndex = "200"
            canvas.style.position = "relative"

            const glContext = canvas.getContext("webgl2")

            if (!glContext) {
                console.log("cant make webgl canvas")
                document.removeChild(canvas)
                return
            }

            const program = makeProgram(glContext)
            if (!program) {
                console.log("cant make webgl program")
                document.removeChild(canvas)
                return
            }

            gl.current = glContext
            gl.current.useProgram(program)
            resize(gl.current, map)
            data.current = generateData()
            setupCirclePattern(gl.current)
            setupCircleVertexs(gl.current, program)
            setupCircleCenter(gl.current, program, data.current)
            const extent = map.getView().calculateExtent()
            //lat,lon
            const bottomLeft: [number, number] = [extent[0], extent[1]]
            //lat,lon
            const topRight: [number, number] = [extent[2], extent[3]]

            render(glContext, program, data.current!.length / 2, bottomLeft, topRight, radius.current)

            const onView = (_e: Event | BaseEvent) => {
                resize(gl.current!, map)
                const extent = map.getView().calculateExtent()
                //lat,lon
                const bottomLeft: [number, number] = [extent[0], extent[1]]
                //lat,lon
                const topRight: [number, number] = [extent[2], extent[3]]
                render(gl.current!, program, data.current!.length / 2, bottomLeft, topRight, radius.current)
            }

            const view = map.getView()
            view.on("change:center", onView)
            view.on("change:resolution", onView)

            gl.current = glContext
            document.getElementsByClassName("ol-viewport")[0].appendChild(canvas)
            return () => {
                document.removeChild(canvas)
                view.un("change:center", onView)
                view.un("change:resolution", onView)
            }
        }
    }, [map])

    return undefined
}